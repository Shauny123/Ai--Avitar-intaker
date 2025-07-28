/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  
  // Internationalization configuration
  i18n: {
    locales: [
      'en', 'en-gb', 'es', 'es-mx', 'fr', 'de', 'it', 'pt-br',
      'zh', 'zh-tw', 'ja', 'ko', 'hi', 'bn', 'ur', 'ar', 'he',
      'ru', 'tr', 'th', 'vi', 'vi-south', 'id', 'tl',
      'sw', 'ha', 'yo', 'zu', 'am', 'bo', 'mn', 'ne', 'el'
    ],
    defaultLocale: 'en',
    localeDetection: true,
  },

  // Image optimization
  images: {
    domains: ['flagcdn.com', 'cdn.jsdelivr.net'],
    formats: ['image/webp', 'image/avif'],
  },

  // Environment variables to expose to client
  env: {
    NEXT_PUBLIC_APP_NAME: process.env.NEXT_PUBLIC_APP_NAME,
    NEXT_PUBLIC_APP_VERSION: process.env.NEXT_PUBLIC_APP_VERSION,
    NEXT_PUBLIC_DEFAULT_LANGUAGE: process.env.NEXT_PUBLIC_DEFAULT_LANGUAGE,
  },

  // Security headers
  async headers() {
    return [
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Content-Security-Policy',
            value: "default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:;",
          },
        ],
      },
    ];
  },

  // Redirects for language variants
  async redirects() {
    return [
      {
        source: '/filipino',
        destination: '/tl',
        permanent: true,
      },
      {
        source: '/tagalog',
        destination: '/tl',
        permanent: true,
      },
      {
        source: '/chinese',
        destination: '/zh',
        permanent: true,
      },
      {
        source: '/arabic',
        destination: '/ar',
        permanent: true,
      },
    ];
  },

  // Webpack configuration for handling multiple languages
  webpack: (config, { isServer }) => {
    // Handle SVG files
    config.module.rules.push({
      test: /\.svg$/,
      use: ['@svgr/webpack'],
    });

    // Handle font files for different languages
    config.module.rules.push({
      test: /\.(woff|woff2|eot|ttf|otf)$/,
      use: {
        loader: 'file-loader',
        options: {
          publicPath: '/_next/static/fonts/',
          outputPath: 'static/fonts/',
        },
      },
    });

    return config;
  },

  // Experimental features
  experimental: {
    serverComponentsExternalPackages: ['@prisma/client'],
  },

  // Output configuration for different deployment targets
  output: process.env.NODE_ENV === 'production' ? 'standalone' : undefined,

  // Compression
  compress: true,

  // Generate source maps in development
  productionBrowserSourceMaps: false,
  
  // Custom page extensions
  pageExtensions: ['ts', 'tsx', 'js', 'jsx'],

  // Power off X-Powered-By header
  poweredByHeader: false,

  // Trailing slash configuration
  trailingSlash: false,
};

module.exports = nextConfig;