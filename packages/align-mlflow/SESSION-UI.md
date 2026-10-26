# Session UI

- **Adds:** readable session titles, action names, and movable, hideable columns in grouped sessions.
- **Install location:** the server; everyone using its UI gets the patch.

**Requires:** Git, `uv`, and Node.js **24.14+ within 24.x** on `PATH`. From the repository root, before starting the server:

```bash
./packages/align-mlflow/scripts/setup.sh
```

- **Default setup:** installs locked MLflow **3.17.2** and the UI together.
- **First build:** downloads dependencies and needs several GB of disk space.

**Reinstall only the UI:** `./packages/align-mlflow/scripts/install-ui.sh`. Stop the server before setup or reinstallation, then restart and reload browsers. See [server setup](SERVER.md#quick-start).

After importing, open **Open World results** or filter the experiment's **Traces** table. Use **Display > Columns** to hide or reorder columns, and **Views** to save a personal layout. [Saved views](#saved-views) are optional.

Views display recorded metadata without changing traces. Older imports can have missing driver or model values. Source version shows the recorded version string; a dirty version does not identify the exact executed code.

Session titles use recorded run, model, scenario, alignment target, and episode tags. Sessions without the required Open World tags display their ID.

## Scenario navigation and evaluation comparisons

In **Traces**, use the filter control in a metadata column header to open MLflow's native filter builder for that field. Choose an operator and a suggested recorded value, or type a value. Suggested values include recorded first-action traces across the experiment, including later result pages. An active header control identifies a filtered field; clearing that field preserves other filters and session selections. The native **Filters** control also supports hidden columns and other fields.

In **Evaluation runs**, select two runs and group by session. Open World comparisons show scenario, alignment target, ADM, model, run, session, and recorded score columns by default. **Episode details** restores these columns when a personal layout hides them. Request and response columns remain available in the column selector.

The grouped comparison pairs episodes by their exact recorded scenario and alignment target when each run has one episode for that pair. Both runs are fetched completely before applying the scenario and target filters, so filtering cannot hide ambiguous metadata. Session IDs and links keep their original values. Different targets, missing or inconsistent metadata, and repeated episodes stay separate; repetitions need a shared trial identifier before they can be paired reliably. A warning identifies ambiguous cases. Other trace-level filters or free-text search disable this pairing because they can hide repetitions. Ungrouped comparisons retain MLflow's native input matching.

This matching and navigation are part of the frontend patch. They do not change source traces, assessments, or run metrics, and do not align individual actions from diverging episodes. Matching assumes the scenario and target IDs identify the same evaluation case across runs; verify the benchmark version before comparing runs from different study revisions.

## Compare sessions and decisions

Use the grouped **Traces** table to check the sessions you want, then click **Compare selected sessions**. **Select sessions in table**, available from **Evaluation runs** and session pages, opens the grouped table. Use **Display > Columns** to choose the metadata and score columns. Selections stay checked across filters and pages; the selection count identifies sessions outside the current page. **Clear session selection** starts over. Session checkboxes select complete sessions by ID even when only some of their traces are on the current page.

Compare two or more sessions without requiring matching scenarios, targets, ADMs, LLMs, or runs. Each pane has a minimum width and the comparison scrolls horizontally, preserving readable content with multiple sessions. Each pane shows its recorded metadata, episode score, independent decision sequence, and the normal session decision sidebar. Clicking a sidebar step selects it and scrolls only that session's conversation to the decision.

Choose decisions from two or three sessions and click **Compare selected decisions** to open MLflow's native trace comparison. **Clear selected decision** removes a pane from that trace comparison without closing its session. Session and decision links retain their original identifiers, and the comparison URL can be reopened or shared with someone who has access to the same server. No corresponding step numbers are required.

For individual traces already visible in the ungrouped **Traces** table, select two or three rows and use MLflow's existing **Actions > Compare**. Session selection and the comparison layout are part of this frontend patch; the decision sidebar, conversation rendering, and trace comparison reuse MLflow's native components.

## Open World results

Open **Open World results** in the experiment's Evaluation navigation. The
dashboard starts with a **Results overview** panel containing run, scenario/target
pair, and episode counts. **Runs** opens
the experiment's native **Model training** runs table. Pair and episode counts
are contextual totals.
Nonzero missing scores, target mismatches, and metadata conflicts link to the
affected sessions. Requested and evaluated targets should agree; a mismatch
indicates a run, scoring, or import problem. **Filter dashboard** uses MLflow's existing
filter controls for search and invocation. Removable chips for applied values
wrap below the totals, with full values available on hover or keyboard focus.
Source revisions remain in run details. The scope panel states that these filters apply to the
run summaries, plot, and matrix. Help icons beside
the counts and section headings explain score averaging, coverage, provenance,
and drill-down behavior.

**Performance by run** summarizes each recorded configuration separately. Its
ADM, model, and run fields use the same labels as the chart and matrix. The
run label omits a repeated leading ADM name. Source revision appears in the
summary; **Open run** links to MLflow's full run details.
The borderless **Mean** action opens the contributing episodes. Its
average gives each scored scenario/target case equal weight after averaging
repeated scored episodes within that case. Missing scores are excluded, never
zero. This differs from the importer's episode-weighted invocation mean when
case coverage or repetition counts differ. Inspect coverage and source revisions
before interpreting differences between runs.

**Episode scores** shows one clickable dot per scored episode and a
diamond for the equal-case mean. The score axis stays fixed from 0 to 1 across
filters. Out-of-range recorded values are marked at the nearest boundary and
retain their raw values in tooltips and summaries. The chart follows the same
dashboard filters as the summaries and matrix. Hover or focus a dot for its score,
scenario, target, ADM, and model; mean tooltips show case and episode coverage.
Nearby dots keep separate positions and larger hit areas; hovering or focusing
a dot highlights it. Tooltips leave neighboring dots reachable and update
immediately when moving between points.
Missing outcomes remain accessible without fabricated chart values. Run, ADM,
LLM, driver, and recorded provenance keep configurations separate. Optional
source-family and scoring-version rows in run summaries are shown only when recorded; the ADM
source revision is not a scorer version.

**Scenario/target matrix** appears below the score distributions with no summary footer.
Labels align left. A single configuration uses a compact table; multiple
configurations use the available width and scroll horizontally when needed.
Select a row heading to show its sessions across runs, or a column heading to
show that configuration's sessions across scenarios. Both selections respect the
dashboard filters and are preserved in the URL. Column headings omit repeated
provenance details.
Ordinary single-episode cells display only the score. Repeated cases show their episode
count; incomplete cases show scored/total counts. Missing scores and absent
episodes remain distinct. Target and metadata conflicts stay visible only when present.

Selecting a review signal, run mean, chart point, or matrix cell scrolls
to its exact contributing sessions in the same grouped session table used by
MLflow's performance view. Session rows retain native names, links, selection,
expandable first-action rows, and resizable columns. The column menu controls
visibility; column choices, widths, and sorting remain when dashboard filters
change. Drilldowns and **Back to results** move keyboard focus to their
destination. Opening a session shows its complete decision sequence. Select
sessions across cells and filters for the existing comparison. This table shows
only the requested target; target mismatches remain visible as warnings.
Normal rows have no metadata-conflict status column.

The **Sessions** table remains visible when filters are cleared. Chart and matrix
selections populate editable conditions in MLflow's native trace filter builder.
A matrix cell supplies invocation, scenario, and target conditions; a dot also
supplies its session. Remove a condition to broaden the results, or use **Show
all dashboard sessions** to clear the table filters. The builder supports the
same fields and operators as the native traces table, including arbitrary tags,
metadata, feedback, and span fields. Matching uses MLflow search across all
of a session's decisions and retains the complete session result in the table.
Active dashboard filters also appear beside the table and can be removed there.
New chart and matrix selections replace the table filters. **Clear checked
sessions** resets comparison checkboxes separately. Dashboard filters, table
filters, and drilldowns are preserved in the URL; checked sessions and column
preferences remain while the dashboard is open.

All data comes from the selected experiment's completed first-action traces and
persisted session assessments. Revisiting or returning focus reloads all result
pages and replaces the complete snapshot atomically. Background loading or errors
retain the last complete results; errors remain visible. The continuously focused
page does not poll. There is no additional database or refresh control. This page
is part of the pinned frontend patch; native charts and trace views remain available.

## Saved views

Use MLflow's **Views** menu to save filters and column layouts for the experiment. Filter column headers or use **Filters** to select ADMs and invocations directly. Saved views are optional personal layouts; server setup and ingestion do not create them.

Saved views belong to the experiment and need no server restart. Opening a saved view restores its filters and columns. Use **Display > Columns** to adjust the layout and save it under a name in **Views**.

MLflow groups the current page of traces into sessions, so a session can span pages and the footer counts traces.
