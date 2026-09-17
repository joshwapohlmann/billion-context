import fs from "node:fs";
import path from "node:path";
import { dumpsDir } from "./paths.js";
import { log as loggerLog } from "./logger.js";

// #762: when the upstream rejects the forwarded body (4xx), persist the exact
// bytes that were sent so the rejection can be explained byte-for-byte. The
// standing body dump (ACP_DUMP_BODY=1) must be armed BEFORE the incident; this
// one fires on the failure itself. Still off by default — conversation bodies
// leak to disk (#276) — enable with BILI_DUMP_4XX=1.
const DEFAULT_MAX_BYTES = 2 * 1024 * 1024;

let failCount = 0;
let lastFailLog = 0;

function warnDumpFailure(err: unknown): void {
    failCount++;
    const now = Date.now();
    if (failCount === 1 || now - lastFailLog >= 60_000) {
        lastFailLog = now;
        const msg = err instanceof Error ? err.message : String(err);
        loggerLog("warn", `[dump] rejected-body dump failed (total ${failCount}x): ${msg}`);
    }
}

/** Write the rejected forwarded body to `<dumpDir>/err-<ts>-<sid>-<status>.json`
 *  when BILI_DUMP_4XX=1. Returns the file path, or null when disabled/skipped/failed. */
export function dumpRejectedBody(status: number, sessionId: string, body: string | Buffer): string | null {
    if (process.env.BILI_DUMP_4XX !== "1") return null;
    const raw = typeof body === "string" ? body : body.toString("utf8");
    if (!raw) return null;
    try {
        const cap = Math.max(1024, Number(process.env.BILI_DUMP_4XX_MAX_BYTES) || DEFAULT_MAX_BYTES);
        let text: string;
        let marker = "";
        if (raw.length > cap) {
            text = raw.slice(0, cap);
            marker = `\n[truncated: ${raw.length - cap} more character(s)]\n`;
        } else {
            try {
                text = JSON.stringify(JSON.parse(raw), null, 2);
            } catch {
                text = raw;
            }
        }
        const dir = dumpsDir();
        fs.mkdirSync(dir, { recursive: true });
        const sid = sessionId.replace(/[^a-zA-Z0-9_-]/g, "_");
        const out = path.join(dir, `err-${Date.now()}-${sid}-${status}.json`);
        fs.writeFileSync(out, `${text}${marker}`);
        loggerLog("info", `[dump] upstream ${status} rejected body written to ${out}`);
        return out;
    } catch (err) {
        warnDumpFailure(err);
        return null;
    }
}
