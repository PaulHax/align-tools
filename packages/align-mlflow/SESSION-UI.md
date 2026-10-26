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

After importing, [install the comparison views](#saved-views) for the experiment. Select **Session comparison** from **Views** and use **Display > Columns** to hide or reorder columns. Save a custom layout as another named view.

Views display recorded metadata without changing traces. Older imports can have missing driver or model values. Source version shows the recorded version string; a dirty version does not identify the exact executed code.

Session titles use recorded run, model, scenario, alignment target, and episode tags. Sessions without the required Open World tags display their ID.

## Scenario navigation and evaluation comparisons

The **Traces** and **Evaluation runs** tables have searchable **Scenario** and **Alignment target** selectors. The **Traces** table also has **ADM**, **LLM**, **Invocation**, and **Source version** selectors. Each selector applies an exact recorded value and keeps other filters. Invocation choices show the run label and identity key. **Reset episode filters** clears these metadata filters while preserving unrelated filters and session selections. Options come from all recorded first-action traces in the experiment or selected runs, including later result pages. Selecting a scenario in **Traces** keeps the current view's other filters, such as its invocation. Clear the selectors to return to the full view.

In **Evaluation runs**, select two runs and group by session. Open World comparisons show scenario, alignment target, ADM, model, run, session, and recorded score columns by default. **Episode details** restores these columns when a personal layout hides them. Request and response columns remain available in the column selector.

The grouped comparison pairs episodes by their exact recorded scenario and alignment target when each run has one episode for that pair. Both runs are fetched completely before applying the scenario and target filters, so filtering cannot hide ambiguous metadata. Session IDs and links keep their original values. Different targets, missing or inconsistent metadata, and repeated episodes stay separate; repetitions need a shared trial identifier before they can be paired reliably. A warning identifies ambiguous cases. Other trace-level filters or free-text search disable this pairing because they can hide repetitions. Ungrouped comparisons retain MLflow's native input matching.

This matching and navigation are part of the frontend patch. They do not change source traces, assessments, or run metrics, and do not align individual actions from diverging episodes. Matching assumes the scenario and target IDs identify the same evaluation case across runs; verify the benchmark version before comparing runs from different study revisions.

## Compare sessions and decisions

Use the grouped **Traces** table to check the sessions you want, then click **Compare selected sessions**. **Select sessions in table**, available from **Evaluation runs** and session pages, opens the existing **ADM performance** table with scenario, alignment target, ADM, LLM, run, and score columns. Selections stay checked across filters and pages; the selection count identifies sessions outside the current page. **Clear session selection** starts over. Session checkboxes select complete sessions by ID even when only some of their traces are on the current page.

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
Normal rows have no metadata-conflict status column. Filters and selected
contributors are preserved in the URL; checkbox selections remain while using
the dashboard.

All data comes from the selected experiment's completed first-action traces and
persisted session assessments. Revisiting or returning focus reloads all result
pages and replaces the complete snapshot atomically. Background loading or errors
retain the last complete results; errors remain visible. The continuously focused
page does not poll. There is no additional database or refresh control. This page
is part of the pinned frontend patch; native charts and trace views remain available.

## Saved views

From the repository root, with the environment installed, point to the destination server and an existing experiment:

```bash
export MLFLOW_TRACKING_URI='http://localhost:5000'
export MLFLOW_EXPERIMENT_NAME='Open World Phase 2'
./packages/align-mlflow/scripts/views.sh "$MLFLOW_TRACKING_URI" --by-adm --by-run
```

The command prints links and installs:

- **Open-world actions:** individual action traces.
- **Episodes by ADM and target:** the first action of each episode.
- **Session comparison:** grouped sessions with scenario, ADM, alignment target, driver, model, and run columns.
- **ADM performance:** one grouped row per episode, with ADM, model, run, scenario, action target, TA3 alignment score, and TA3 evaluated target. The first-action filter keeps each episode's recorded feedback with its row. Open the session to inspect all its decisions.
- **ADM: &lt;label&gt;:** one Session comparison view per distinct nonempty `adm` run tag, with an exact ADM filter.
- **Run: &lt;label&gt; [&lt;key&gt;]:** one Session comparison view per imported invocation. The exact `align.run_key` filter keeps invocations separate even when their readable names match.

All views use **All time**, so older runs remain visible. Choose an **ADM:** or **Run:** entry in **Views** to filter without typing. Labels come from the destination experiment. Both flags are optional; omit them to install only the standard presets.

Rerun after importing new ADMs or invocations. Existing named presets are restored, new labels get views, and other views are retained, including views for labels no longer present. Unchanged views are left untouched. Save personal layouts under another name before rerunning; conflicting names or IDs stop setup before any views change. Saved views belong to the experiment and need no server restart.

Definitions live in [views.json](examples/views.json); [configure_views.py](examples/configure_views.py) generates the ADM and invocation variants. The recipe uses MLflow's internal saved-view format and checks for **3.17.2** before writing. Use the destination's UI URL as the first argument if it differs from `MLFLOW_TRACKING_URI`; standard MLflow credentials apply.

The performance columns require this package's session UI patch. Episode scores are session feedback, not scores for the individual action that stores them. Missing feedback is unavailable, not zero. The evaluated target comes from that feedback and should match the action target; a mismatch indicates a run, scoring, or import problem. When comparing ADMs, use the same scenario/target cases and account for repeated episodes and missing results.

Saved views provide filter shortcuts. MLflow still groups the current page of traces into sessions, so a session can span pages and the footer counts traces.
