import fs from "fs"
import path from "path"

const TEMP_DIR = path.join(process.cwd(), "temp")
export function ensureTempDir() {
  if (!fs.existsSync(TEMP_DIR)) {
    fs.mkdirSync(TEMP_DIR, { recursive: true })
  }
}
ensureTempDir()
