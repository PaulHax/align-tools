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
sessions. **Runs** opens the experiment's **Model training** runs table.

**Performance by run** separates recorded configurations by run, ADM, model,
driver, and provenance. **Open run** shows full run details; the **scored episodes**
count opens the contributing sessions. The mean gives each scored scenario/target
pair equal weight, averaging repeated scored episodes within each pair first.
Missing scores are excluded, never treated as zero. This differs from the
importer's [episode-weighted run mean](INGESTION.md#run-summary-scores) when
repetition counts differ. Check coverage and source revisions before comparing runs.

**Episode scores** shows a clickable dot for each scored episode and a diamond
for the mean. Hover or focus a point for its score and metadata. The axis is fixed
from 0 to 1; out-of-range scores appear at the nearest boundary, with their raw
values in tooltips and summaries. Missing scores remain accessible through the
overview and session table.

The **Scenario/target matrix** shows scores by configuration and evaluation case.
Repeated episodes show a count; incomplete results show scored/total counts.
Missing scores and absent episodes are distinct. Select a row for that case
across runs, a column for that configuration, or a cell for its contributing
sessions.

Chart points, matrix selections, and review links filter the **Sessions (episodes)**
table below. Edit conditions from a column header or **Filters**, remove a
condition to broaden results, or use **Clear filters**. New chart and matrix
selections replace the table filters. Filters and drilldowns are preserved in
the URL. **Back to results** returns to the summary.

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

## Compare sessions and decisions

In the grouped **Traces** table or **Sessions (episodes)**, check sessions and
click **Compare selected sessions**. **Select sessions in table**, available from
**Evaluation runs** and session pages, opens the grouped table. Checkboxes select
complete sessions, including decisions outside the current page. Selections
remain across filters and pages; a notice identifies checked sessions outside
the current table.

Compare two or more sessions with any scenarios, targets, ADMs, models, or runs.
Each pane shows recorded metadata, episode score, and its own decision sequence.
The comparison scrolls horizontally when needed. Clicking a sidebar step selects
and scrolls to that session's decision.

Choose decisions from two or three sessions, then click **Compare selected
decisions**. **Clear selected decision** removes a decision from that comparison
without closing its session. Decisions need no matching step numbers. Comparison
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
