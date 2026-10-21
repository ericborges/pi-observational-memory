import { describe, expect, it } from "vitest";

import { normalizeSourceEntryIds, OBSERVATION_TIMESTAMP_PATTERN, ObserverStreamError, parseTextModePayload, runObserver } from "../src/agents/observer/agent.js";
import { AGENT_LOOP_MAX_TOKENS } from "../src/model-budget.js";

function fakeAgentLoop(handler: (prompts: any[], context: any, config: any) => Promise<void> | void, events: any[] = []): any {
	return ((prompts: any[], context: any, config: any) => ({
		async *[Symbol.asyncIterator]() {
			for (const event of events) yield event;
		},
		result: async () => {
			await handler(prompts, context, config);
			return {};
		},
	})) as any;
}

function assistantEndEvent(stopReason: string, errorMessage?: string): any {
	return { type: "message_end", message: { role: "assistant", stopReason, errorMessage } };
}

describe("runObserver maxTokens clamping", () => {
	function captureLoopConfig() {
		let loopConfig: any;
		const loop = fakeAgentLoop((_prompts, _context, config) => {
			loopConfig = config;
		});
		return { loop, config: () => loopConfig };
	}

	const args = {
		apiKey: "test",
		priorReflections: [],
		priorObservations: [],
		chunk: "[Source entry id: entry-a]\nUser asked for a memory update.",
		allowedSourceEntryIds: ["entry-a"],
	};

	it("clamps the loop maxTokens to a model whose maxTokens is below the configured budget", async () => {
		const { loop, config } = captureLoopConfig();

		await runObserver({
			...args,
			model: { maxTokens: 8_192 } as any,
			maxOutputTokens: 32_000,
			agentLoop: loop,
		});

		expect(config().maxTokens).toBe(8_192);
	});

	it("passes the configured maxOutputTokens through when the model advertises no maxTokens", async () => {
		const { loop, config } = captureLoopConfig();

		await runObserver({
			...args,
			model: {} as any,
			maxOutputTokens: 8_192,
			agentLoop: loop,
		});

		expect(config().maxTokens).toBe(8_192);
	});

	it("defaults the loop maxTokens to AGENT_LOOP_MAX_TOKENS", async () => {
		const { loop, config } = captureLoopConfig();

		await runObserver({ ...args, model: {} as any, agentLoop: loop });

		expect(config().maxTokens).toBe(AGENT_LOOP_MAX_TOKENS);
	});
});

describe("OBSERVATION_TIMESTAMP_PATTERN", () => {
	it("matches local minute timestamps without regex shorthand escapes", () => {
		expect(OBSERVATION_TIMESTAMP_PATTERN).not.toContain("\\d");
		const pattern = new RegExp(OBSERVATION_TIMESTAMP_PATTERN);
		expect(pattern.test("2026-05-02 10:30")).toBe(true);
		expect(pattern.test("2026-5-02 10:30")).toBe(false);
		expect(pattern.test("2026-05-02T10:30")).toBe(false);
		expect(pattern.test("2026-05-02 10:30:00")).toBe(false);
	});
});

describe("runObserver", () => {
	const baseArgs = {
		model: {} as any,
		apiKey: "test",
		priorReflections: [],
		priorObservations: [],
		chunk: "[Source entry id: entry-a]\nUser asked for a memory update.",
		allowedSourceEntryIds: ["entry-a"],
	};

	it("keeps core observer prompt rules", async () => {
		let systemPrompt = "";
		const loop = fakeAgentLoop((_prompts, context) => {
			systemPrompt = context.messages[0]?.role === "system" ? context.messages[0].content : "";
		});

		await runObserver({ ...baseArgs, agentLoop: loop });

		expect(systemPrompt).toContain("Preserve user assertions exactly");
		expect(systemPrompt).toContain("Detail preservation");
		expect(systemPrompt).toContain("Frame state changes as supersession");
		expect(systemPrompt).toContain("sourceEntryIds");
		expect(systemPrompt).toContain("zero observations");
		expect(systemPrompt).toContain("The dropper will drop these first");
		expect(systemPrompt).toContain("highest-resistance, load-bearing observations");
		expect(systemPrompt).not.toContain("will NEVER be dropped");
		expect(systemPrompt).not.toContain("pruner");
	});

	it("records V3 observations with source ids and code-computed tokenCount", async () => {
		const content = "User asked for a memory update.";
		const loop = fakeAgentLoop(async (_prompts, context) => {
			await context.tools[0].execute("tool-1", {
				observations: [{ timestamp: "2026-05-02 10:30", content, relevance: "high", sourceEntryIds: ["entry-a"] }],
			});
		});

		const observations = await runObserver({ ...baseArgs, agentLoop: loop });

		expect(observations).toHaveLength(1);
		expect(observations?.[0]).toMatchObject({
			content,
			timestamp: "2026-05-02 10:30",
			relevance: "high",
			sourceEntryIds: ["entry-a"],
			// tokenCount is code-computed from the full rendered line (id + timestamp + relevance + content).
			tokenCount: 18,
		});
		expect(observations?.[0].id).toMatch(/^[a-f0-9]{12}$/);
	});

	it("rejects invented source ids and returns no observations", async () => {
		const loop = fakeAgentLoop(async (_prompts, context) => {
			await context.tools[0].execute("tool-1", {
				observations: [{ timestamp: "2026-05-02 10:30", content: "Bad source", relevance: "medium", sourceEntryIds: ["missing"] }],
			});
		});

		await expect(runObserver({ ...baseArgs, agentLoop: loop })).resolves.toBeUndefined();
	});

	it("dedupes deterministic ids", async () => {
		const loop = fakeAgentLoop(async (_prompts, context) => {
			await context.tools[0].execute("tool-1", {
				observations: [
					{ timestamp: "2026-05-02 10:30", content: "Same content", relevance: "medium", sourceEntryIds: ["entry-a"] },
					{ timestamp: "2026-05-02 10:31", content: "Same content", relevance: "high", sourceEntryIds: ["entry-a"] },
				],
			});
		});

		const observations = await runObserver({ ...baseArgs, agentLoop: loop });

		expect(observations).toHaveLength(1);
		expect(observations?.[0].content).toBe("Same content");
	});

	it("returns undefined when no tool call records observations", async () => {
		const loop = fakeAgentLoop(() => {});
		await expect(runObserver({ ...baseArgs, agentLoop: loop })).resolves.toBeUndefined();
	});

	it("throws ObserverStreamError when the stream errors with nothing recorded", async () => {
		for (const stopReason of ["error", "aborted"]) {
			const loop = fakeAgentLoop(() => {}, [assistantEndEvent(stopReason, "prompt is too long")]);
			const error = await runObserver({ ...baseArgs, agentLoop: loop }).catch((e) => e);
			expect(error).toBeInstanceOf(ObserverStreamError);
			expect(error.stopReason).toBe(stopReason);
			expect(error.message).toContain("prompt is too long");
		}
	});

	it("keeps partial observations when the stream errors after recording", async () => {
		const loop = fakeAgentLoop(async (_prompts, context) => {
			await context.tools[0].execute("tool-1", {
				observations: [{ timestamp: "2026-05-02 10:30", content: "Kept despite later error", relevance: "high", sourceEntryIds: ["entry-a"] }],
			});
		}, [assistantEndEvent("error", "gateway timeout")]);

		const observations = await runObserver({ ...baseArgs, agentLoop: loop });

		expect(observations).toHaveLength(1);
		expect(observations?.[0].content).toBe("Kept despite later error");
	});

	it("uses finishTurn as an observer turn cap without overriding hard exits", async () => {
		let finishTurn: any;
		const loop = fakeAgentLoop((_prompts, _context, config) => {
			finishTurn = config.finishTurn;
		});

		await runObserver({ ...baseArgs, agentLoop: loop, maxTurns: 2 });

		expect(finishTurn).toBeTypeOf("function");
		expect(finishTurn({ message: { stopReason: "error" } })).toBeUndefined();
		expect(finishTurn({ message: { stopReason: "aborted" } })).toBeUndefined();
		expect(finishTurn({ message: { stopReason: "toolUse" } })).toBeUndefined();
		expect(finishTurn({ message: { stopReason: "stop" } })).toEqual({ action: "end" });
	});

	it("uses configured observer thinking level for reasoning models", async () => {
		let seenReasoning: unknown;
		const loop = fakeAgentLoop((_prompts, _context, config) => {
			seenReasoning = config.reasoning;
		});

		await runObserver({ ...baseArgs, model: { reasoning: true } as any, agentLoop: loop, thinkingLevel: "minimal" });

		expect(seenReasoning).toBe("minimal");
	});

	it("omits observer reasoning when thinkingLevel is off", async () => {
		let seenReasoning: unknown = "unset";
		const loop = fakeAgentLoop((_prompts, _context, config) => {
			seenReasoning = config.reasoning;
		});

		await runObserver({ ...baseArgs, model: { reasoning: true } as any, agentLoop: loop, thinkingLevel: "off" });

		expect(seenReasoning).toBeUndefined();
	});
});

describe("parseTextModePayload", () => {
	const validItem = { timestamp: "2026-05-02 10:30", content: "A fact.", relevance: "medium", sourceEntryIds: ["entry-a"] };

	it("parses wrapped, bare-array, and bare-single-object shapes", () => {
		expect(parseTextModePayload(JSON.stringify({ observations: [validItem] }))).toEqual({ observations: [validItem] });
		expect(parseTextModePayload(JSON.stringify([validItem]))).toEqual({ observations: [validItem] });
		expect(parseTextModePayload(JSON.stringify(validItem))).toEqual({ observations: [validItem] });
	});

	it("tolerates surrounding whitespace", () => {
		expect(parseTextModePayload(`  ${JSON.stringify([validItem])}\n`)).toEqual({ observations: [validItem] });
	});

	it("drops shape-invalid items, keeps shape-valid ones (pool check happens in execute)", () => {
		const payload = [
			validItem,
			{ ...validItem, content: "Bad relevance", relevance: "urgent" },
			{ ...validItem, content: "Bad timestamp", timestamp: "2026-05-02T10:30" },
			{ ...validItem, content: "No content", content: "" },
			{ ...validItem, content: "Bad source type", sourceEntryIds: "entry-a" },
			"not-an-object",
			{ ...validItem, content: "Invented id", sourceEntryIds: ["not-in-the-chunk"] },
		];
		const parsed = parseTextModePayload(JSON.stringify(payload));
		expect(parsed).toEqual({ observations: [validItem, payload[6]] });
	});

	it("returns undefined for empty, non-JSON, and all-invalid input", () => {
		expect(parseTextModePayload("")).toBeUndefined();
		expect(parseTextModePayload("   ")).toBeUndefined();
		expect(parseTextModePayload("Nothing new to record.")) .toBeUndefined();
		expect(parseTextModePayload("{ not json")).toBeUndefined();
		expect(parseTextModePayload(JSON.stringify({ observations: ["nope"] }))).toBeUndefined();
		expect(parseTextModePayload(JSON.stringify(42))).toBeUndefined();
	});
});

describe("runObserver text-mode recovery", () => {
	const baseArgs = {
		model: {} as any,
		apiKey: "test",
		priorReflections: [],
		priorObservations: [],
		chunk: "[Source entry id: entry-a]\nUser asked for a memory update.",
		allowedSourceEntryIds: ["entry-a"],
	};

	function textLoop(text: string): any {
		return fakeAgentLoop(() => {}, [{ type: "message_end", message: { role: "assistant", stopReason: "stop", content: [{ type: "text", text }] } }]);
	}

	it("recovers a wrapped payload written as plain text, with tool-path-identical records", async () => {
		const content = "User asked for a memory update.";
		const loop = textLoop(JSON.stringify({ observations: [{ timestamp: "2026-05-02 10:30", content, relevance: "high", sourceEntryIds: ["entry-a"] }] }));

		const observations = await runObserver({ ...baseArgs, agentLoop: loop });

		expect(observations).toHaveLength(1);
		expect(observations?.[0]).toMatchObject({ content, timestamp: "2026-05-02 10:30", relevance: "high", sourceEntryIds: ["entry-a"] });
		expect(observations?.[0].tokenCount).toBe(18);
		expect(observations?.[0].id).toMatch(/^[a-f0-9]{12}$/);
	});

	it("recovers a bare-array payload", async () => {
		const loop = textLoop(JSON.stringify([{ timestamp: "2026-05-02 10:30", content: "Bare array fact.", relevance: "medium", sourceEntryIds: ["entry-a"] }]));

		const observations = await runObserver({ ...baseArgs, agentLoop: loop });

		expect(observations).toHaveLength(1);
		expect(observations?.[0].content).toBe("Bare array fact.");
	});

	it("rejects a text payload whose source ids are not in the allowed pool", async () => {
		const loop = textLoop(JSON.stringify({ observations: [{ timestamp: "2026-05-02 10:30", content: "Bad source", relevance: "medium", sourceEntryIds: ["missing"] }] }));
		await expect(runObserver({ ...baseArgs, agentLoop: loop })).resolves.toBeUndefined();
	});

	it("treats a short plain-text confirmation as a deliberate empty", async () => {
		const loop = textLoop("Nothing new to record in this chunk.");
		await expect(runObserver({ ...baseArgs, agentLoop: loop })).resolves.toBeUndefined();
	});

	it("returns undefined for long non-JSON text", async () => {
		const loop = textLoop("This is a long paragraph of prose about the conversation that is not a payload. ".repeat(5));
		await expect(runObserver({ ...baseArgs, agentLoop: loop })).resolves.toBeUndefined();
	});

	it("keeps tool-call records and does not double-count a later text payload", async () => {
		const loop = fakeAgentLoop(
			async (_prompts, context) => {
				await context.tools[0].execute("tool-1", {
					observations: [{ timestamp: "2026-05-02 10:30", content: "From tool", relevance: "high", sourceEntryIds: ["entry-a"] }],
				});
			},
			[{ type: "message_end", message: { role: "assistant", stopReason: "stop", content: [{ type: "text", text: JSON.stringify({ observations: [{ timestamp: "2026-05-02 10:31", content: "From text", relevance: "high", sourceEntryIds: ["entry-a"] }] }) }] } }],
		);

		const observations = await runObserver({ ...baseArgs, agentLoop: loop });

		expect(observations).toHaveLength(1);
		expect(observations?.[0].content).toBe("From tool");
	});

	it("still throws ObserverStreamError on stream failure even when text is present", async () => {
		const loop = fakeAgentLoop(() => {}, [
			{ type: "message_end", message: { role: "assistant", stopReason: "error", errorMessage: "boom", content: [{ type: "text", text: JSON.stringify({ observations: [{ timestamp: "2026-05-02 10:30", content: "Lost", relevance: "high", sourceEntryIds: ["entry-a"] }] }) }] } },
		]);

		const error = await runObserver({ ...baseArgs, agentLoop: loop }).catch((e) => e);
		expect(error).toBeInstanceOf(ObserverStreamError);
	});

	it("recovers from a string-typed content message", async () => {
		const loop = fakeAgentLoop(() => {}, [
			{ type: "message_end", message: { role: "assistant", stopReason: "stop", content: JSON.stringify(["nope"]) } },
		]);
		await expect(runObserver({ ...baseArgs, agentLoop: loop })).resolves.toBeUndefined();

		const loop2 = fakeAgentLoop(() => {}, [
			{ type: "message_end", message: { role: "assistant", stopReason: "stop", content: JSON.stringify({ observations: [{ timestamp: "2026-05-02 10:30", content: "String content fact.", relevance: "low", sourceEntryIds: ["entry-a"] }] }) } },
		]);
		const observations = await runObserver({ ...baseArgs, agentLoop: loop2 });
		expect(observations).toHaveLength(1);
		expect(observations?.[0].content).toBe("String content fact.");
	});
});

describe("normalizeSourceEntryIds", () => {
	const allowed = ["entry-a", "entry-b", "entry-c"];

	it("accepts source ids from the allowed chunk and orders them by branch order", () => {
		expect(normalizeSourceEntryIds(["entry-c", "entry-a"], allowed)).toEqual(["entry-a", "entry-c"]);
	});

	it("dedupes repeated source ids", () => {
		expect(normalizeSourceEntryIds(["entry-b", "entry-b", "entry-a"], allowed)).toEqual(["entry-a", "entry-b"]);
	});

	it("rejects missing, empty, or hallucinated source ids", () => {
		expect(normalizeSourceEntryIds(undefined, allowed)).toBeUndefined();
		expect(normalizeSourceEntryIds([], allowed)).toBeUndefined();
		expect(normalizeSourceEntryIds(["entry-a", "not-in-the-chunk"], allowed)).toBeUndefined();
		expect(normalizeSourceEntryIds(["entry-a"], [])).toBeUndefined();
	});
});
