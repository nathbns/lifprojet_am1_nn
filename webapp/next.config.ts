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
      {
        protocol: 'https',
        hostname: 'nathbns-yolo3-from-scratch.hf.space',
        port: '',
        pathname: '/gradio_api/file=/**',
      },
      {
        protocol: 'https',
        hostname: 'nathbns-preprocess-yoco.hf.space',
        port: '',
        pathname: '/gradio_api/file=/**',
      },
      {
        protocol: 'https',
        hostname: 'lichess1.org',
        port: '',
        pathname: '/assets/**',
      },
    ],
  },
};

export default nextConfig;
