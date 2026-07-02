import { defineConfig } from 'astro/config';

export default defineConfig({
  site: 'https://awaygu.github.io',
  base: '/',
  output: 'static',
  trailingSlash: 'always',
  build: {
    format: 'directory'
  },
  markdown: {
    shikiConfig: {
      theme: 'github-dark',
      wrap: true
    }
  }
});
