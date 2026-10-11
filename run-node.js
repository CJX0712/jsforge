/*
 * run-node.js — 无头自检运行器
 * 依次加载 4 个 forge 模块，运行各自 selftest()，汇总 8/8 全绿状态。
 * 全绿退出码 0；任一失败退出码 1。
 * 用法：node run-node.js
 * 作者：晨星
 */
'use strict';
const mods = ['pca-forge', 'gmm-forge', 'nn-forge', 'attn-forge'];
let allPass = true;
let totalPass = 0;
let totalAll = 0;
const pad = (s, n) => (s + '').padEnd(n);

console.log('══════════════════════════════════════════════');
console.log('  jsforge · 无头自检 · 零依赖 · 作者 晨星');
console.log('══════════════════════════════════════════════');

for (const m of mods) {
  const api = require('./' + m + '.js');
  const res = api.selftest();
  const ok = res.pass === res.total;
  if (!ok) allPass = false;
  totalPass += res.pass;
  totalAll += res.total;
  console.log(`\n[${ok ? '✅' : '❌'}] ${pad(m, 14)} ${res.pass}/${res.total} 全绿`);
  for (const c of res.checks) {
    console.log(`      ${c.pass ? '✅' : '❌'} ${pad(c.name, 28)} ${c.info}`);
  }
}

console.log('\n────────────────────────────────────────────────');
console.log(`  总计: ${totalPass}/${totalAll}  ${allPass ? '✅ 全部通过' : '❌ 存在失败'}`);
console.log('────────────────────────────────────────────────');
process.exit(allPass ? 0 : 1);
