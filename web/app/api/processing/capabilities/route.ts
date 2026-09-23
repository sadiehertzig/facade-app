// GET /api/processing/capabilities: is the PhotoGuard worker ready right now?
import { capabilities, json } from "@/lib/processor";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET() {
  return json(await capabilities());
}
