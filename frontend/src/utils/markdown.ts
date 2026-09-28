// Minimal markdown-to-HTML renderer for assistant answers.
// Ported from src/api/ui.html formatAnswerHTML: HTML is escaped first, then
// markdown transforms run, so the only HTML in the output is generated here.
export function formatAnswerHTML(text: string): string {
  if (!text) return ''
  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
  // Headers
  html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>')
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>')
  // Tables (simple pipe tables with a |---|---| separator row)
  html = html.replace(/((?:^\|.+\|$\n?)+)/gm, (match: string) => {
    const lines = match.trim().split('\n').filter((l) => l.trim())
    if (lines.length < 2) return match
    const separator = lines[1]
    if (!/^\|[\s:|-]+\|$/.test(separator.trim())) return match
    const headerCells = lines[0]
      .split('|')
      .filter((c) => c.trim())
      .map((c) => c.trim())
    const bodyLines = lines.slice(2)
    let table = '<table><thead><tr>'
    headerCells.forEach((c) => (table += `<th>${c}</th>`))
    table += '</tr></thead><tbody>'
    bodyLines.forEach((line) => {
      const cells = line
        .split('|')
        .filter((c) => c.trim())
        .map((c) => c.trim())
      table += '<tr>'
      cells.forEach((c) => (table += `<td>${c}</td>`))
      table += '</tr>'
    })
    table += '</tbody></table>'
    return table
  })
  // Ordered lists
  html = html.replace(/^(\d+)\. (.+)$/gm, '<li>$2</li>')
  html = html.replace(/(<li>.+<\/li>\n?)+/g, (match: string) => '<ol>' + match + '</ol>')
  // Unordered lists
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>')
  html = html.replace(/(<li>.+<\/li>\n?)+/g, (match: string) => {
    if (match.includes('<ol>')) return match
    return '<ul>' + match + '</ul>'
  })
  // Paragraphs (double newlines), then single newlines inside a paragraph
  html = html.replace(/\n\n/g, '</p><p>')
  html = html.replace(/\n/g, '<br>')
  return `<p>${html}</p>`
}
