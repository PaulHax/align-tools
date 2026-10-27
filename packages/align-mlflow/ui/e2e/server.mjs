import { createReadStream } from "node:fs";
import { access, realpath, stat } from "node:fs/promises";
import { createServer } from "node:http";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repository = fileURLToPath(new URL("../../../../", import.meta.url));
const installedBuild = () => {
  const result = spawnSync(
    "uv",
    [
      "run",
      "--project",
      repository,
      "--no-sync",
      "python",
      "-c",
      'from importlib.metadata import distribution; print(distribution("mlflow").locate_file("mlflow/server/js/build"))',
    ],
    {
      cwd: repository,
      encoding: "utf8",
      env: { ...process.env, PYTHONDONTWRITEBYTECODE: "1" },
    },
  );
  if (result.status !== 0)
    throw new Error(
      "Install the patched UI first or set MLFLOW_UI_BUILD_DIR.\n" +
        result.stderr,
    );
  return result.stdout.trim();
};
const directory = await realpath(
  process.env.MLFLOW_UI_BUILD_DIR || installedBuild(),
);
await access(path.join(directory, "index.html"));
const mime = {
  ".html": "text/html",
  ".js": "application/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".gif": "image/gif",
  ".ico": "image/x-icon",
  ".woff": "font/woff",
  ".woff2": "font/woff2",
  ".map": "application/json",
};
const server = createServer(async (request, response) => {
  try {
    if (!["GET", "HEAD"].includes(request.method)) {
      response.writeHead(405).end();
      return;
    }
    const url = new URL(request.url, "http://127.0.0.1:4173");
    if (url.pathname === "/__health") {
      response.writeHead(200, { "Content-Type": "text/plain" }).end("ready");
      return;
    }
    const relative =
      url.pathname === "/"
        ? "index.html"
        : decodeURIComponent(url.pathname)
            .replace(/^\/static-files\//, "")
            .replace(/^\//, "");
    const filename = await realpath(path.resolve(directory, relative));
    if (filename !== directory && !filename.startsWith(directory + path.sep)) {
      response.writeHead(403).end();
      return;
    }
    const info = await stat(filename);
    if (!info.isFile()) {
      response.writeHead(404).end();
      return;
    }
    response.writeHead(200, {
      "Content-Type":
        mime[path.extname(filename)] || "application/octet-stream",
      "Content-Length": info.size,
    });
    if (request.method === "HEAD") response.end();
    else createReadStream(filename).pipe(response);
  } catch (error) {
    response.writeHead(error.code === "ENOENT" ? 404 : 400).end();
  }
});
server.listen(4173, "127.0.0.1");
