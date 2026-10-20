// [Role] Record one accounted live product catalog observation before a step.
// [Invariants] DSH's supported agent/pre-step boundary runs after the profile
// is mounted and before request headers, streams, or automatic session titles.
// [Timeouts & Locks] The caller bounds the probe and drains its process group;
// this hook uses no timers, registry polling, or independent lock.
// [Side-Effect Policy] Write the catalog once; an EXIT=1 probe exits before
// next(), while a normal trial continues the same agent step after the write.
import { writeFileSync } from 'node:fs'

const NAME = 'bench-catalog-checkpoint'

async function capture(ctx, exitAfter) {
  const payload = { schema: 'dsh-live-catalog-checkpoint/2', pid: process.pid, cwd: process.cwd(), exit_after: exitAfter }
  try {
    payload.tools = ctx.tools.schemas().map((tool) => ({ name: tool.name, description: tool.description }))
  } catch (error) {
    payload.tools_error = String(error)
  }
  try {
    const skills = await ctx.skills.list({ cwd: process.cwd() })
    payload.skills = skills.map((skill) => ({
      name: skill.name,
      provider: skill.provider,
      description: skill.description,
    }))
  } catch (error) {
    payload.skills_error = String(error)
  }
  try {
    const rows = []
    for (const entry of ctx.loader.entries()) {
      rows.push({ id: entry.options?.id ?? null, name: entry.options?.name ?? null })
    }
    payload.loader_rows = rows
  } catch (error) {
    payload.loader_error = String(error)
  }
  return payload
}

export function apply(ctx) {
  const output = process.env.MINDIE_BENCH_CATALOG_CHECKPOINT_OUT
  if (!output) return
  const exitAfter = process.env.MINDIE_BENCH_CATALOG_CHECKPOINT_EXIT === '1'
  let written = false
  // Automatic session titles schedule on the later llm/stream/request-header
  // path. Waiting for this pre-step capture prevents their requests from
  // racing catalog observation, including while skills.list() is pending.
  ctx.on('agent/pre-step', async (event, next) => {
    if (!written) {
      written = true
      const payload = await capture(ctx, exitAfter)
      writeFileSync(output, JSON.stringify(payload, null, 2) + '\n')
      if (exitAfter) process.exit(0)
    }
    return next()
  })
}

export const name = NAME
export const inject = ['tools', 'skills']
export default { name: NAME, inject: ['tools', 'skills'], apply }
