import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'nathbns-yolo1-from-scratch.hf.space',
        port: '',
        pathname: '/gradio_api/file=/**',
      },
    ],
  },
};

export default nextConfig;
