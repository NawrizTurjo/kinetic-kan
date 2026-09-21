# Prompt: Produce the Project Final Report

You are producing the final written report for this project. Treat this as a
professional deliverable, not a status update — assume the reader is a
supervisor or external reviewer who was not in the room for any of the work
and knows nothing about this repo's internal structure, file layout, branch
names, or terminology.

## 0. Before writing anything: build real understanding

Do not start drafting from vague memory of "what this project is about."
Read, in this order:

1. **Every finding document** in the project's docs/notes directory, in
   numeric/chronological order if numbered — these are the primary source of
   truth for what was actually done and found.
2. **The project's own presentation or pitch deck**, if one exists — it
   usually states the research question, the base paper being extended, and
   the specific design choices being interrogated more crisply than the
   working notes do.
3. **The base paper or reference paper** this project builds on or compares
   against (find the actual PDF/arXiv source, not just a link list) — you
   need its actual claims, actual numbers, and actual method, not a
   secondhand paraphrase.
4. **The actual code** for each experiment — read the scripts that produced
   the results, not just the docs describing them, so you can verify every
   number and every method description against what the code actually does.
5. **The actual saved output**: metrics files, tables, and every candidate
   figure on disk. Open and look at the images — do not caption a figure you
   have not actually viewed.

If anything in the write-ups is ambiguous, contradictory, or looks like it
might be stale relative to the code, resolve it by reading the code and data
directly before writing a single sentence that depends on it. Never invent a
number, a file, a citation, or a result. If something is genuinely unverifiable
or incomplete, say so explicitly in the report rather than smoothing over it.

## 1. Structure: a paper, not a project-management document

Organize the report the way a technical paper is organized, by what each
section *is*, never by internal project labels:

1. **Executive summary** — one page, headline results only, written last.
2. **Introduction** — the actual scientific/engineering problem being
   addressed, the relevant background theory (derive it — do not just wave
   at it), what the base paper/prior work established, and precisely which
   of its assumptions or design choices this project treats as variables to
   interrogate rather than fixed facts. End with a roadmap sentence.
3. **Methodology / baseline** — how the baseline was reproduced, the
   evaluation protocol used throughout (define every metric here, once, in
   plain language, before it is used anywhere else), and why it's a valid
   protocol.
4. **One section per independent experiment or study**, each self-contained,
   in the order the work was actually done. Chain each one's opening
   sentence to what the *previous* section already established, so the
   report reads as a continuing investigation, not a stapled-together set of
   unrelated reports.
5. **Validation / open work** — anything designed but not yet executed,
   stated honestly as incomplete, with what it would take to close it.
6. **Discussion** — the patterns that recur *across* the independent
   studies. This is not a bullet-point restatement of each section's
   findings; it is new synthesis that could not have been written from any
   single section alone.
7. **Conclusion** — short. What was learned, what remains open.

**Never name a section, subsection, figure caption, or any reader-facing
text after an internal project label** (a phase number, a lettered track, a
branch name, an internal codename). Title every section by what it actually
investigates. Internal labels are for your own bookkeeping while writing,
never for the document a reader sees.

## 2. Every experiment section follows the same scaffold

Before any data or numbers, state — visually set apart from the surrounding
prose, e.g. in a distinctly styled callout box, not just another paragraph —
three things in this order:

- **Question**: what specific thing this experiment is trying to find out.
- **Expected**: what the hypothesis or prior reasoning predicted, stated as
  a real, falsifiable prediction, not a vague "we wanted to see."
- **Observed**: what actually happened, in one or two sentences, plainly
  stating whether the expectation held or was refuted.

Only after that scaffold do you present the setup, the data, the figures,
and the full explanation. Close every experiment section with a second,
visually distinct callout summarizing the verdict — the takeaway a reader
who only skims callout boxes should walk away with.

This is not optional decoration: a reader should be able to flip through the
document reading only the boxed callouts and come away with an accurate
summary of the entire project.

## 3. Writing style

- **Professional and academic, but not stiff.** Write like a competent
  researcher explaining their own work to a peer, not like a press release
  and not like a lab notebook.
- **Explain mechanisms, not just outcomes.** Don't just report a number —
  explain briefly *why* the method being used works the way it does, when
  that mechanism isn't obvious (e.g., derive or state the actual equation
  behind a technique before using it, explain why a particular statistical
  test was chosen over an obvious alternative, explain what a measured
  quantity physically means). Depth here is what separates a report from a
  results dump. Do not, however, pad with irrelevant textbook background —
  every piece of theory included must be load-bearing for something the
  report actually claims or does.
- **No AI-generated-text tics.** Do not use the "it's not X, it's Y"
  construction, "not only X but Y," "X rather than Y" as a rhetorical hedge,
  "not X but rather Y," or any variant of asserting a claim by negating its
  opposite. State claims directly and positively. Do not open findings with
  throat-clearing ("It's worth noting that…", "Interestingly…"). Avoid
  formulaic hedge-then-claim-then-hedge paragraph shapes.
- **No internal file paths, script names, directory structures, CLI flags,
  or code-artifact names in the prose.** A reader should never see a
  backslash-escaped filename, a `--flag`, or a repo path. Describe every
  mechanism, tool, or intervention in plain language instead (a technique's
  actual name or a clear description of what it does, not the flag that
  turns it on).
- **No unexplained abbreviations or shorthand, anywhere, including inside
  tables.** If there is room to write the full word, write the full word —
  "Extrapolation," not "Extrap."; "Training MSE," not "Train MSE"; spell out
  acronyms at first use and define what they mean in plain terms. Do not
  assume the reader already knows this project's internal vocabulary or
  its codebase's jargon.
- **Do not assume prior context.** Every system, every metric, and every
  method should be understandable to someone who has read only this report
  and nothing else — no "as we know," no unexplained proper nouns.

## 4. Math

Include the real mathematics wherever it clarifies a mechanism: governing
equations of every system studied, the actual formula behind any measured
statistic or transform, derivations of any theoretical prediction the data
is later checked against. Keep it purposeful — this should read as
math-supported writing, not a wall of equations for their own sake. If a
piece of math doesn't change what the reader understands about *why*
something happened, cut it.

## 5. Figures

- **Every figure that supports a claim belongs inline, in the section that
  makes that claim**, at the point in the prose where it's actually being
  discussed — never in a separate appendix the reader has to flip to, and
  never referenced only by filename without being shown.
- Select figures by what a reader actually needs to see to believe the
  claim: one flagship summary figure per major result, plus the two or
  three specific supporting panels the prose explicitly leans on (e.g. a
  side-by-side comparison that makes a contrast visually obvious). Do not
  include every generated image — large batches of near-duplicate outputs
  (every random seed, every parameter sweep value) should be represented by
  one or two representative examples, chosen for what they best illustrate,
  not by exhaustiveness.
- Write a caption for every figure that tells the reader what to look for
  and what conclusion to draw from it, not just what the axes are.
- If a finding is fundamentally visual (a shape, a spatial pattern, a
  contrast between two conditions), do not describe it in prose alone —
  show it, and use the prose to point at specific, verifiable features of
  the image.

## 6. Tables

Full descriptive column headers and row labels, not abbreviations. Use
consistent, properly typeset scientific notation. Bold the standout result
per table. Add a units row or units in the header, never leave units
implicit. If a table needs a legend or a note about how a number was
computed, put it directly below the table, not buried in prose elsewhere.

## 7. Verification pass — do this before calling it done

- Cross-check every number in the report against the actual saved output
  file it claims to come from.
- Open and actually look at every figure referenced, and confirm the
  caption's claim about it is visible in the image itself, not just
  plausible.
- Grep your own draft for banned constructions (the negation-hedge patterns
  above), leftover abbreviations, and any stray file path or code-artifact
  mention, and fix every hit.
- Read the document once straight through as a skeptical outside reviewer:
  does every section's "Expected" actually get resolved by its "Observed"?
  Does the Discussion say something the individual sections didn't already
  say? Is any section thin enough that a reader would ask "is that really
  all there is to say about this?" — if so, go find the missing mechanism,
  caveat, or connecting idea and add it.
- Compile the document and confirm it builds cleanly with no errors, and
  no more than trivial layout warnings.

## 8. Deliverable

A single, well-organized source tree for the report — a clean master
document plus one file per section, a shared style/template file so the
whole document looks like one coherent design system, and a script that
recompiles the document automatically on every save so it's fast to iterate.
Produce a final compiled document, and report back what was verified,
what (if anything) remains genuinely open or unverifiable, and why.
