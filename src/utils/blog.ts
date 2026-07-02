import type { CollectionEntry } from 'astro:content';

/** 分类 id → 中文名称，单一数据源 */
export const categoryNames: Record<string, string> = {
  'ai-application': 'AI 应用',
  'engineering': '工程实践',
  'interview': '面试复盘',
};

/** 分类 id → 描述文案 */
export const categoryDescs: Record<string, string> = {
  'ai-application': 'LangChain、RAG、Agent、Prompt Engineering 等技术实践',
  'engineering': '架构选型、部署、性能优化、工具链',
  'interview': '面试记录、问题总结、知识盲点补齐',
};

/** 估算阅读时长（按 400 字/分钟取整，至少 1 分钟） */
export function estimateReadingTime(text: string): number {
  return Math.max(1, Math.ceil((text || '').length / 400));
}

/** 日期格式化为 yyyy.MM.dd */
export function fmtDate(d: Date): string {
  return d
    .toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' })
    .replace(/\//g, '.');
}

/** 取文章 slug 的末段（兼容 ai-application/xxx 形式） */
export function postSlug(post: CollectionEntry<'blog'>): string {
  return post.slug.split('/').pop() || post.slug;
}

/** 构造文章详情页 URL */
export function postUrl(post: CollectionEntry<'blog'>, base = '/'): string {
  return `${base}blog/${post.data.category}/${postSlug(post)}/`;
}

/** 仅保留已发布文章并按日期倒序排列 */
export function getPublishedPosts(posts: CollectionEntry<'blog'>[]): CollectionEntry<'blog'>[] {
  return posts
    .filter(post => !post.data.draft)
    .sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

/** 从已发布文章中聚合实际存在文章的分类列表 */
export function getUsedCategories(posts: CollectionEntry<'blog'>[]) {
  const published = getPublishedPosts(posts);
  return [...new Set(published.map(p => p.data.category))].map(id => ({
    id,
    name: categoryNames[id] || id,
  }));
}
