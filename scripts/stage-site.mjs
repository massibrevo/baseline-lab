// Sites accepts a top-level static root. Copy only the already-built public web output.
import { cpSync, existsSync, rmSync } from 'node:fs';
if (!existsSync('apps/web/dist/index.html')) throw new Error('Run npm run build --prefix apps/web first');
rmSync('dist', { recursive: true, force: true });
cpSync('apps/web/dist', 'dist', { recursive: true });
console.log('Staged static public demo in dist/');
