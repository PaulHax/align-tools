# Session UI

Browse Open World results, filter sessions, and compare episodes and decisions.
Install on each MLflow server; everyone using that server gets these features.

**Requires:** Git, `uv`, and Node.js **24.14+ within 24.x** on `PATH`.
Stop the server, then run from the repository root:

```bash
./packages/align-mlflow/scripts/setup.sh
```

Setup installs locked MLflow **3.17.2** and all the UI features described here.
The first build downloads dependencies and needs several GB of disk space.
Restart the server and reload browsers afterward. To reinstall only the UI,
run `./packages/align-mlflow/scripts/install-ui.sh` while the server is stopped.
See [server setup](SERVER.md#quick-start).

## Open World results

Open **Open World results** in the experiment's Evaluation navigation.
**Results overview** shows run, scenario/target pair, and episode counts.
Missing scores, target mismatches, and metadata conflicts link to affected
sessions. **Runs** opens the experiment's **Evaluation runs** table, including
recorded final score metrics. Use its columns and comparisons to inspect runs.

**Results overview**, **Episode scores**, **Scenario/target matrix**, and
**Sessions (episodes)** expand or collapse when you select their heading or left
chevron. Expanded sections use the page's main vertical scroll. Wide tables and
plots retain horizontal scrolling.
An information icon beside each heading explains its contents. Sessions start
collapsed and open when a chart, matrix, or review link selects episodes.

The plot mean gives each scored scenario and evaluated target pair equal weight,
averaging repeated scored episodes within each pair first.
The evaluated target comes from the recorded TA3 assessment; requested action
targets are retained separately. Configurations explicitly recorded as unaligned
are labelled **Unaligned**. An absent requested target alone does not establish
the alignment mode. A mean is unavailable when a scored episode lacks a recorded
scenario or evaluated target; its raw score remains available for inspection.

Missing scores are excluded, never treated as zero. This differs from the
importer's [episode-weighted run mean](INGESTION.md#run-summary-scores) when
repetition counts differ. Check coverage and source revisions before comparing runs.

**Episode scores** shows clickable score markers and a diamond for the mean.
Episodes with exactly equal scores share a counted marker; selecting it opens
all contributing episodes. Nearby distinct scores remain separate. Hover or focus
a marker for its score and metadata. The axis is fixed
from 0 to 1; out-of-range scores appear at the nearest boundary, with their raw
values in tooltips and summaries. Tooltips show a matching evaluated and requested
target once, with separate requested context for mismatched, unaligned, or missing
targets. Missing scores remain accessible through the overview and session table.
Vertical positions only separate nearby markers and stay stable when the plot
width changes; horizontal positions follow the score axis.

The **Scenario/target matrix** shows scores by configuration, scenario, and
**Evaluated target**. Scored episodes use the target reported by TA3, including
unaligned configurations that were scored against that target. Unscored episodes
appear in separate rows labelled **Requested target**; they are not assigned an
inferred evaluated target. Run columns lead with the recorded ADM implementation
name and wrap it within a readable column width. Hover or focus a case for its
full IDs, or use a run's information button for its configuration, provenance,
and run details link. Case labels stay visible while scrolling horizontally.
Repeated episodes show a count; incomplete results show scored/total counts.
Missing scores and absent episodes are distinct. Select a row for that case
across runs, a column for that configuration, or a cell for its contributing
sessions.

Chart points, matrix selections, and review links filter the **Sessions (episodes)**
table below. Edit conditions from a column header or **Filters**, remove a
condition to broaden results, or use **Clear filters**. New chart and matrix
selections replace the table filters. Filters and drilldowns are preserved in
the URL.

The table supports the same filtering, column controls, and comparisons as
**Traces**. A matching decision includes its complete session, so opening or
expanding it shows the full decision history. Check sessions across filters for
comparison; **Clear checked sessions** resets those selections separately.
Selections stay within the current experiment.

Results refresh when you revisit the page or return focus. The page does not
poll continuously. Loading or errors retain the last complete results, with
errors shown on the page.

## Filters and sorting

In **Traces** or **Sessions (episodes)**, use a column-header filter or **Filters**
to select runs, scenarios, targets, and other recorded fields. Suggested values
include episodes outside the current filters and result page. Clearing one
field preserves other filters and checked sessions.

Use a column's header menu or **Display > Sort** to sort. Text sorts
alphabetically, numeric fields numerically, and missing values stay last.
Sorting applies before pagination and keeps session groups together.
In **Traces**, the pagination footer counts sessions when grouped, or rows when
standalone traces are also present.

Use **Display > Columns** to hide or reorder columns. Column choices, widths,
and sorting remain when filters change.

## Saved views

In **Traces**, use **Views** to save filters and column layouts for the experiment. Opening a
saved view restores its filters and columns. Views are optional personal layouts;
setup and ingestion do not create them, and saving one needs no server restart.

## Inspect a decision

Each decision has one outer frame without nested field outlines. Compact source
captions identify the JSON sections shown beneath the readable summaries:
`input_output.json → input` and `input_output.json → output`. Spacing and a
divider distinguish the situation from the action. Code keeps its preformatted
shading.
Each situation groups the current patients into **Now nearby**, **Not nearby**, and
**Unknown proximity**. Patient updates focus on clinical and description changes
instead of repeating proximity changes and unchanged descriptions.
**Decision process** follows Situation and precedes the chosen action. Each stage
has a single-line preview of its action, recommendation, or recorded rationale.
The decision sidebar contains only decisions. Expand a stage to read its recorded
proposal, advice, or response alongside the relevant settings and measurements.
The original activity payloads and source links appear below the readable
content when the stage opens. The Decision process and Supporting evidence
headings use the same native typography.

Reviewer proposals appear with **Proposal**, advice and computed suggestions with
**Review**, and the final action and whether guidance was followed with **Final
selection**. **Review settings** holds influence, swap threshold, and the review
contract; **Review measurements** holds confidence, estimated improvement, and
strength. Explicitly owned metadata appears with its relevant stage.

The decision timeline combines stage summaries with captured agent messages,
thinking, and tool calls under **Decision process**. **Proposal** and **Review**
follow a uniquely matched proposal call with a recorded action review marker.
With uncertain stage timing, **Saved stage summaries** follows the recorded
activity. **Final selection** closes the reviewed timeline. Decisions with only
stage summaries show those stages in order.

Stages, thinking, and tool calls use the same compact disclosure row. Icons
and labels identify the row type. **Agent message** rows use readable Markdown.
**Tool call** rows show the tool name and key arguments.
Open a parsed call to inspect its arguments, any matched result, and the
**Raw log excerpt** control. When a matched observation log contains no return
value, the row shows **Result not captured**. Calls with malformed arguments
open directly to the original excerpt. Recorded thinking has its own expandable
row.

**Supporting evidence** contains setup logs and other source records.
**Capture gaps** explains limitations of the available evidence. The collapsed
rows preview their contents. Opening **Source logs** shows the excerpt labels,
complete available excerpts, and source links. Opening **Other source evidence**
shows its original payloads and links. **Choice info** shows the complete
original `choice_info` value from the decision record. The output source
retains the saved environment action.
Original records and source references remain available for inspection. Cache
hits remain distinct from fresh model invocations.

The input and output source previews show two lines of code with a single
**See more** control for the complete JSON. Expanded code has **See less** at the top and
bottom, so collapsing a long payload does not require scrolling to its end.
Expanded Situation and action summaries also have a collapse control at the top.
Readable summaries and source previews use matching chevron expansion buttons.
Session code views omit line-number gutters.
Single-object previews omit their outer braces and excess indentation; arrays,
nested objects, and raw log text retain their structure. The copy control still
copies complete, valid JSON for structured data.
Disclosure labels use the same bold styling. Generic log rows include a response
excerpt or a tool name and arguments instead of repeating uninformative headings
and `recorded` badges. The complete source remains available when the row opens.
Saved records show the fields they contain. Missing request or result notices
apply to captured model and tool activities, where those fields are expected.
Process payloads use their exact recorded JSON fields or source-log ranges.
Opening an activity loads its selected log range on demand; opening the page or
a phase does not fetch those payloads. **Open source artifact** provides the
complete captured log. Large payloads above the inline limit remain available
through their source artifact.

The chevron at the right of the session metrics row collapses and expands
**Session assessments**. The same control remains in place while collapsed, and
the panel gives its full width back to the trace. Decision selection, resized
panel width, and unsaved notes are retained.

Older imports can gain process evidence with the [card refresh command](INGESTION.md#refresh-existing-decision-evidence).

## Compare sessions and decisions

In the grouped **Traces** table or **Sessions (episodes)**, check sessions and
click **Compare selected sessions**. **Select sessions in table**, available from
**Evaluation runs** and session pages, opens the grouped table. Checkboxes select
complete sessions, including decisions outside the current page. Selections
remain across filters and pages; a notice identifies checked sessions outside
the current table.

Compare two or more sessions with any scenarios, targets, ADMs, models, or runs.
Each pane shows recorded metadata, episode score, and its own decision sequence.
The comparison scrolls horizontally when needed. Single-session and comparison
pages share the decision viewer, navigation, and selection behavior. Clicking a
sidebar step selects and scrolls to that session's decision.

Choose decisions from two or three sessions, then click **Compare selected
decisions**. Select another decision to replace a session's choice. Decisions
need no matching step numbers. Comparison
URLs can be reopened or shared with people who have access to the same server.
For individual traces in ungrouped **Traces**, select two or three rows and use
**Actions > Compare**.

## Evaluation run comparisons

In **Evaluation runs**, select two runs and group by session. Open World
comparisons show scenario, target, ADM, model, run, session, and score columns.
**Episode details** restores these columns if a personal layout hides them.
Request and response columns remain available in the column selector.

Episodes pair by their exact recorded scenario and alignment target only when
each run has one episode for that pair. Different targets, missing or inconsistent
metadata, and repeated episodes stay separate, with a warning for ambiguous cases.
Scenario and target filters cannot hide repetitions to make a pair appear unique.
Other trace-level filters or free-text search disable this pairing because they
can hide repetitions. Ungrouped comparisons use MLflow's input matching.

Pairing does not align individual actions from diverging episodes. Verify that
scenario and target IDs identify the same evaluation case when comparing runs
from different study revisions.
