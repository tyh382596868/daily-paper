---
name: daily-code-teach
description: |
  Daily code teaching note writer (Step 2 of 2 pipeline). Reads candidates from
  /tmp/daily_code_candidates.json, fetches the actual code, writes 2 educational
  markdown notes (one from tracked repos, one from a trending project), updates
  INDEX.md and topic indexes.

  Triggers: "write daily code notes", "跑一下 daily code 笔记", debugging Step 2 only.
---

> **Before starting**: Say "📝 Writing today's code lessons" and announce today's date.

# Daily Code Teach

You are the educational note writer. Take candidates from Step 1, fetch the actual
source code, and produce 2 high-quality teaching notes in English.

## Prerequisites

Check that `/tmp/daily_code_candidates.json` exists. If not, tell the user to run
`/daily-code-fetch` first and stop.

## Step 1: Fetch the actual code

For each candidate (`tracked` and `trending`):

1. Use the cached clone from `{cache_dir}/{repo_name}` (Step 1 already cloned them).
2. Read the file specified by `file` and the line range `lines`.
3. If `lines` only narrows to a function/class boundary approximately, expand to the
   nearest logical unit (don't cut mid-function).
4. Strip excessive comments only if they obscure the teaching point; otherwise keep
   the original code verbatim.

## Step 2: Write the teaching note

Use this **exact template** for each note. Output language: English.

```markdown
---
date: YYYY-MM-DD
topic: robotics | diffusion | infrastructure
source: tracked | trending
repo: owner/name
file: path/to/file.py
permalink: https://github.com/.../blob/{sha}/path#L20-L95
difficulty: beginner | intermediate | advanced
read_time: ~10 min
tags: [code-of-the-day, {topic}, {specific-technique}]
---

# {Concept Title}

> **In one line**: {one-sentence summary of what you'll learn}

## Why this matters

{2-3 sentences. Concrete payoff: what does understanding this unlock for the reader?
Avoid generic motivation. Tie it to a real problem the reader probably faces.}

## The code

`{repo}` — [`{file}`]({permalink})

\`\`\`python
{the actual code, verbatim or lightly trimmed}
\`\`\`

## What's happening

{Step-by-step walkthrough. Use numbered list. Each step should reference a specific
line number or code construct. Don't paraphrase — explain the WHY.}

1. **Line N (`some_call(...)`)**: ...
2. **Lines M-K (the loop)**: ...
3. **The return value**: ...

## The analogy

{One concrete mental model from outside the code domain. Example: "Action tokenization
is like assigning Morse code to robot joints — you're picking a vocabulary the LLM
already knows how to spell with." Must be vivid, not vague. Bad: "it's like a function
that does X". Good: "imagine a sushi conveyor belt where..."}

## Try it yourself

The minimum runnable example that demonstrates the same idea, free of the original
codebase's dependencies:

\`\`\`python
# < 30 lines, runs as-is with pip install of 1-2 common deps
{minimal_example}
\`\`\`

Run with:
\`\`\`bash
pip install {deps}
python try.py
\`\`\`

Expected output:
\`\`\`
{what you should see}
\`\`\`

## Where this pattern shows up elsewhere

- **{Other repo 1}**: {how they use the same idea}
- **{Other repo 2}**: {variation/improvement}
- **{Related paper / blog}**: {link if known}

## Caveats / when it breaks

{1-3 honest limitations. When does this NOT work? What's the next thing the reader
should learn after this?}

## Further reading

- [{Link 1}]({url})
- [{Link 2}]({url})
```

### Quality bar (mandatory)

A note is acceptable ONLY if all of these are true:

1. **Code is real** — verbatim or lightly trimmed from the actual file, not paraphrased.
2. **Permalink works** — pinned to commit SHA, not branch name.
3. **Analogy is concrete** — names a specific physical/everyday object or scenario.
4. **Try-it example runs** — fewer than 30 lines, declares its deps, no proprietary
   data needed.
5. **Length** — 150-400 lines total. Less = too thin. More = lost the teaching point.

If a candidate cannot meet these (e.g. the code is too tangled, or you can't write a
real analogy), **swap to another candidate** rather than ship a weak note. Tell the user
which one you swapped to and why.

## Step 3: Save and index

For each note, save to:
```
DailyCode/{YYYY}/{MM}/{YYYY-MM-DD}-{slug}.md
```

`{slug}` format: `{source-tag}-{concept-kebab-case}`. Examples:
- `openvla-action-tokenizer.md` (tracked source: use repo short name)
- `trending-{repo-shortname}-{concept}.md` (trending source: prefix with "trending")

Then update:

### INDEX.md
Prepend the new entry at the top (newest first) under a `## Archive` section:

```markdown
| Date       | Topic       | Title                                       | Source   |
|------------|-------------|---------------------------------------------|----------|
| 2026-05-10 | robotics    | [Action tokenizer (OpenVLA)](2026/05/...md) | tracked  |
| 2026-05-10 | robotics    | [Discovered: {repo}'s ...](2026/05/...md)   | trending |
```

### README.md
Update the `<!-- auto-updated by daily-code-teach -->` section under `## Latest` with
the 5 most recent entries (linked).

### topics/{topic}.md
Append the new entry to the relevant topic index file. If the file doesn't exist, create
it with this header:

```markdown
# {Topic title}

Notes tagged `{topic}`, newest first.

| Date | Title | Repo |
|------|-------|------|
```

## Step 4: Commit (optional)

If the user said anything about pushing or committing (e.g. "and push it"), then:
1. Stage all new/modified files in `DailyCode/`
2. Commit with message: `daily code: YYYY-MM-DD {topic} ({tracked-repo} + {trending-repo})`
3. Push to current branch

Otherwise, do NOT commit — just tell the user the files are ready and they can review
before committing.

## Output

Tell the user:
- 2 notes written: list paths
- INDEX.md, README.md, topic indexes updated
- (If committed) commit hash and branch

## Notes

- **Do NOT skip the trending note** even if the tracked one took a long time.
- **Do NOT auto-translate to Chinese** — output is English-only by spec.
- If GitHub API rate-limits you (search for trending), fall back to a manual search or
  re-use a recently-discovered project from the cache.
