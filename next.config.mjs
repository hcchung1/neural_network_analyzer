/** @type {import('next').NextConfig} */
const sidecarBaseUrl = (
  process.env.PYTHON_SIDECAR_URL || "http://127.0.0.1:8080/internal/v1"
).replace(/\/$/, "");

const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: "/internal/v1/:path*",
        destination: `${sidecarBaseUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
