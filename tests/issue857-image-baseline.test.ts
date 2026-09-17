import assert from "node:assert/strict";
import http from "node:http";
import { once } from "node:events";
import test from "node:test";

process.env.NODE_ENV = "test";

import { defaultConfig } from "acp-kernel";
import { startServer, type ProxyOptions } from "../src/server.ts";
import { SessionStore, _setStoreForTest } from "../src/persist.ts";
import { _setForTest as setRegistryForTest } from "../src/registry.ts";
import { listSessions } from "../src/session.ts";

// Issue #857: on non-pixel-billing upstreams (DeepSeek et al.), the
// conservative bytes-mode image estimate (base64/4) overestimates real image
// billing by ~100× — and that estimate leaked into the usage baseline
// (preflight write-back, armFailureShrink, weak-overflow arming), then
// masqueraded as upstream-accepted evidence: the upward window self-heal
// CONFIRMED the inflated number as the model's real window, and the #496
// escape hatch stayed permanently closed because "baseline ≥ window + a
// learned limit" read as overflow evidence. A DeepSeek session with
// screenshots then 502-loops forever despite ~50K real input.
//
// Fix under test:
//   C1 preflight write-back persists text+overhead only (image floor excluded)
//   C2 stats.lastInputTokensSource tags every baseline raise ("usage"|"estimate")
//   C3 upward self-heal + stale-limit retraction trust "usage" baselines only
//      (absent on legacy files = untrusted)
//   C4 #496 hatch: an untrusted baseline is not overflow evidence, and a
//      learned limit ABOVE the configured window is residue, not evidence
//
// Conventions copied from tests/issue767-pixel-billing.test.ts (mock
// Responses upstream + startServer e2e). Windows/baselines mirror the issue's
// session: window 1,000,000, fake confirmed 1,072,519.

const WINDOW = 1_000_000;
const FAKE_CONFIRMED = 1_072_519;
const MODEL = "deepseek-flash";
// No route/table/launcher match → the window resolves purely from the global
// fallback, so nativeFromFallback=true and the upward self-heal may fire.
const FALLBACK_MODEL = "unknown-model-857";
// One screenshot whose BYTES-mode cost (b64/4) is exactly IMAGE_FLOOR tokens —
// large enough to exceed even the fake 1,072,519 window on its own, as in the
// issue (multiple pages of screenshots).
const IMAGE_FLOOR = 1_100_000;
// pngB64() emits 32 base64 characters of header; bytes cost = ceil(len/4).
const IMG_PAD = IMAGE_FLOOR * 4 - 32;

function pngB64(w: number, h: number, padChars: number): string {
    const b = Buffer.alloc(24);
    b.set([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a], 0);
    b.writeUInt32BE(13, 8);
    b.write("IHDR", 12, "ascii");
    b.writeUInt32BE(w, 16);
    b.writeUInt32BE(h, 20);
    return b.toString("base64") + "A".repeat(padChars);
}

interface MockStats { streamingForwards: number; imagesSeen: number; summaryCalls: number }

function startMockUpstream(): Promise<{ server: http.Server; port: number; stats: MockStats }> {
    const stats: MockStats = { streamingForwards: 0, imagesSeen: 0, summaryCalls: 0 };
    const server = http.createServer((req, res) => {
        const chunks: Buffer[] = [];
        req.on("data", (c: Buffer) => chunks.push(c));
        req.on("end", () => {
            const raw = Buffer.concat(chunks).toString("utf8");
            let parsed: { stream?: boolean } = {};
            try { parsed = JSON.parse(raw); } catch { parsed = {}; }
            if (parsed.stream === false) {
                stats.summaryCalls += 1;
                res.writeHead(200, { "content-type": "application/json" });
                res.end(JSON.stringify({ output_text: "PREFLIGHT SUMMARY: folded segment." }));
                return;
            }
            stats.streamingForwards += 1;
            stats.imagesSeen += (raw.match(/"input_image"/g) ?? []).length;
            res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-cache" });
            res.write(`event: response.completed\ndata: ${JSON.stringify({ response: { id: "resp_done", status: "completed", output: [], usage: { input_tokens: 6617, output_tokens: 5, total_tokens: 6622 } } })}\n\n`);
            res.end();
        });
    });
    server.listen(0, "127.0.0.1");
    return new Promise((resolve) => {
        server.on("listening", () => resolve({ server, port: (server.address() as { port: number }).port, stats }));
    });
}

async function startProxy(upstreamPort: number, models: Record<string, { context: number }>, modelContextLimit: number): Promise<{ proxy: http.Server; port: number }> {
    _setStoreForTest(new SessionStore({ enabled: false }));
    setRegistryForTest({});
    const proxy = await startServer({
        port: 0,
        host: "127.0.0.1",
        upstream: "http://127.0.0.1",
        routes: { [`http://127.0.0.1:${upstreamPort}`]: { models } },
        modelContextLimit,
        kernelConfig: defaultConfig(modelContextLimit),
        compress: { injectTool: true, injectNudge: true },
        promptCache: { routing: "auto" },
        sessionHeader: "x-acp-session",
        log: false,
        debug: false,
        passthrough: false,
        autoUpdate: false,
        mitm: { enabled: false, domains: [] },
    } as ProxyOptions);
    await once(proxy, "listening");
    return { proxy, port: (proxy.address() as { port: number }).port };
}

function body(model: string, sessionId: string, input: unknown[]): string {
    return JSON.stringify({ model, stream: true, store: false, session_id: sessionId, instructions: "You are the test coding agent.", input, max_output_tokens: 1024 });
}

function imageInput(): unknown[] {
    return [{ type: "message", role: "user", content: [{ type: "input_text", text: "screenshot attached" }, { type: "input_image", image_url: `data:image/png;base64,${pngB64(1653, 2339, IMG_PAD)}` }] }];
}

function fillerHistory(n: number): unknown[] {
    const out: unknown[] = [];
    for (let i = 0; i < n; i++) out.push({ type: "message", role: i % 2 ? "assistant" : "user", content: `MARKER_${i}: ${"lorem ipsum dolor sit amet ".repeat(40)}` });
    return out;
}

function sess(label: string) {
    const s = listSessions().find((x) => x.meta.label === label);
    assert.ok(s, `session ${label} established`);
    return s!;
}

function confirmedOf(label: string): Record<string, number> | undefined {
    return sess(label).metadata["confirmedContextLimits"] as Record<string, number> | undefined;
}

test("#857 A: legacy poisoned baseline + fake confirmed window — image turn is released by the #496 hatch and recovers to real usage", async () => {
    const { server: upstream, port: uport, stats } = await startMockUpstream();
    const { proxy, port } = await startProxy(uport, { [MODEL]: { context: WINDOW } }, WINDOW);
    try {
        const url = `http://127.0.0.1:${port}/bili/http://127.0.0.1:${uport}/v1/responses`;
        const headers = { "content-type": "application/json" };

        const r1 = await fetch(url, { method: "POST", headers, body: body(MODEL, "s857a", [{ type: "message", role: "user", content: "hello there" }]) });
        assert.equal(r1.status, 200);
        await r1.text();

        // Reproduce the issue's stuck state: an estimate-derived baseline the
        // OLD code self-healed into a confirmed window. Both values sit above
        // the real 1M window. Turn 1's usage report already stamped a provenance
        // flag — delete it to simulate a legacy persisted file (pre-#857), which
        // carries no lastInputTokensSource at all.
        const s = sess("s857a");
        s.stats.lastInputTokens = FAKE_CONFIRMED;
        delete s.stats.lastInputTokensSource;
        (s.metadata as Record<string, unknown>).confirmedContextLimits = { [MODEL]: FAKE_CONFIRMED };

        const r2 = await fetch(url, { method: "POST", headers, body: body(MODEL, "s857a", imageInput()) });
        assert.equal(r2.status, 200, "the issue's stuck state forwards instead of 502-looping");
        await r2.text();
        assert.equal(stats.streamingForwards, 2);
        assert.equal(stats.imagesSeen, 1, "screenshot reached the upstream verbatim");

        const after = sess("s857a");
        assert.ok(after.stats.lastInputTokens < WINDOW, `baseline recovered to real usage (got ${after.stats.lastInputTokens})`);
        assert.equal(after.stats.lastInputTokensSource, "usage");
        assert.deepEqual(confirmedOf("s857a"), { [MODEL]: FAKE_CONFIRMED }, "non-governing residue is left in place but inert");

        const r3 = await fetch(url, { method: "POST", headers, body: body(MODEL, "s857a", imageInput()) });
        assert.equal(r3.status, 200, "steady state keeps working after recovery");
        await r3.text();
        assert.equal(stats.streamingForwards, 3);
        assert.equal(stats.imagesSeen, 2);
    } finally {
        upstream.close();
        proxy.close();
    }
});

test("#857 B: genuine overflow evidence (learned limit == configured window) still hard-fails — byte-relay protection intact", async () => {
    const { server: upstream, port: uport, stats } = await startMockUpstream();
    const { proxy, port } = await startProxy(uport, { [MODEL]: { context: WINDOW } }, WINDOW);
    try {
        const url = `http://127.0.0.1:${port}/bili/http://127.0.0.1:${uport}/v1/responses`;
        const headers = { "content-type": "application/json" };

        const r1 = await fetch(url, { method: "POST", headers, body: body(MODEL, "s857b", [{ type: "message", role: "user", content: "hello there" }]) });
        assert.equal(r1.status, 200);
        await r1.text();

        // A byte-counting relay that REJECTED at the configured window is real
        // evidence: confirmed limit == window, stale baseline on top. Legacy
        // file → no provenance field (turn 1's usage report stamped one).
        const s = sess("s857b");
        s.stats.lastInputTokens = FAKE_CONFIRMED;
        delete s.stats.lastInputTokensSource;
        (s.metadata as Record<string, unknown>).confirmedContextLimits = { [MODEL]: WINDOW };

        const r2 = await fetch(url, { method: "POST", headers, body: body(MODEL, "s857b", imageInput()) });
        assert.equal(r2.status, 502, "no blind forward when the upstream proved the window");
        await r2.text();
        assert.equal(stats.streamingForwards, 1, "image turn was not forwarded");
        assert.equal(stats.imagesSeen, 0);
    } finally {
        upstream.close();
        proxy.close();
    }
});

test("#857 C: usage-grounded over-window baseline — hatch stays closed AND the fold write-back cannot re-poison the baseline with the image floor", async () => {
    const { server: upstream, port: uport, stats } = await startMockUpstream();
    // No route model entry: the window is a pure fallback, so the upward
    // self-heal (which requires nativeFromFallback) can confirm it.
    const { proxy, port } = await startProxy(uport, {}, WINDOW);
    try {
        const url = `http://127.0.0.1:${port}/bili/http://127.0.0.1:${uport}/v1/responses`;
        const headers = { "content-type": "application/json" };

        const r1 = await fetch(url, { method: "POST", headers, body: body(FALLBACK_MODEL, "s857c", [{ type: "message", role: "user", content: "hello there" }]) });
        assert.equal(r1.status, 200);
        await r1.text();

        // A REAL usage report put the baseline above the window (legitimate
        // self-heal evidence) — but the image floor exceeds even the healed
        // window, so compression cannot make the payload fit.
        const s = sess("s857c");
        s.stats.lastInputTokens = FAKE_CONFIRMED;
        s.stats.lastInputTokensSource = "usage";

        // Filler history gives the kernel foldable ranges so the write-back
        // path (C1) actually runs during the exhausted preflight.
        const r2 = await fetch(url, { method: "POST", headers, body: body(FALLBACK_MODEL, "s857c", [...fillerHistory(16), ...imageInput()]) });
        assert.equal(r2.status, 502, "usage evidence keeps the hatch closed");
        await r2.text();
        assert.equal(stats.streamingForwards, 1);
        assert.ok(stats.summaryCalls > 0, "ranges were folded (write-back path exercised)");

        const after = sess("s857c");
        assert.ok(after.stats.lastInputTokens < WINDOW + 100_000, `baseline not re-poisoned by the image floor (got ${after.stats.lastInputTokens})`);
        assert.equal(after.stats.lastInputTokensSource, "usage", "grounded provenance preserved across the fold");
        assert.deepEqual(confirmedOf("s857c"), { [FALLBACK_MODEL]: FAKE_CONFIRMED }, "usage-derived baseline confirms the window");
    } finally {
        upstream.close();
        proxy.close();
    }
});

test("#857 D: upward self-heal trusts usage baselines only (estimate blocked, usage confirmed)", async () => {
    const { server: upstream, port: uport } = await startMockUpstream();
    // No route model entry: "deepseek-flash" resolves via CONTEXT_LIMIT_TABLE
    // (/^deepseek/i → 1M since #852) — a fallback source, so nativeFromFallback=true.
    const { proxy, port } = await startProxy(uport, {}, 200_000);
    try {
        const url = `http://127.0.0.1:${port}/bili/http://127.0.0.1:${uport}/v1/responses`;
        const headers = { "content-type": "application/json" };

        const r1 = await fetch(url, { method: "POST", headers, body: body(MODEL, "s857d", [{ type: "message", role: "user", content: "hello there" }]) });
        assert.equal(r1.status, 200);
        await r1.text();

        let s = sess("s857d");
        s.stats.lastInputTokens = 1_100_000;
        s.stats.lastInputTokensSource = "estimate";
        const r2 = await fetch(url, { method: "POST", headers, body: body(MODEL, "s857d", [{ type: "message", role: "user", content: "second turn" }]) });
        assert.equal(r2.status, 200);
        await r2.text();
        s = sess("s857d");
        assert.equal(confirmedOf("s857d"), undefined, "an estimate-derived baseline must not confirm a window");

        // Turn 2's successful usage report reset the baseline — re-stamp it.
        s.stats.lastInputTokens = 1_100_000;
        s.stats.lastInputTokensSource = "usage";
        const r3 = await fetch(url, { method: "POST", headers, body: body(MODEL, "s857d", [{ type: "message", role: "user", content: "third turn" }]) });
        assert.equal(r3.status, 200);
        await r3.text();
        s = sess("s857d");
        assert.deepEqual(confirmedOf("s857d"), { [MODEL]: 1_100_000 }, "a usage-derived baseline confirms the window");
    } finally {
        upstream.close();
        proxy.close();
    }
});
