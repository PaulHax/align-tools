# UI integration

- Add a custom component only when the existing components cannot support the required behavior. Keep the custom part small and compose it with the native components.
- Reuse shared renderers, interaction logic, and state wiring across single-session and comparison views so equivalent actions behave consistently.
- Fix behavior in the shared path and remove superseded rendering or interaction infrastructure when consolidating views.

# Trace and session presentation

- Omit repetitive boilerplate from summaries and labels when it adds no distinction between decisions or activities, such as a `recorded` suffix on every activity.
- Make activity labels specific enough to scan: include the tool or action name, relevant arguments, or a brief excerpt rather than repeating generic producer headings.
- Keep shared configuration at the run level or within the relevant settings disclosure. Distinguish it from measurements and outcomes that change per decision.
- Count workflow stages separately from source logs, capture notes, and saved records. Put known metadata with its owning stage and retain unassigned evidence separately.
- Include a concise, source-backed preview in collapsed stage and evidence rows; an activity count alone does not explain their contents.
- Use MLflow's native Typography components and theme tokens. Keep a small type scale with one consistent font size, weight, line height, and color per information role; peer headings must use identical styles. Avoid ad hoc fonts, size changes, and bold treatments.
- Use a consistent visual hierarchy and expansion style. Opening a workflow stage or source group must reveal all of its contents without nested activity, settings, or payload expansion controls.
- Preserve original source data and metadata for inspection; simplify their presentation rather than altering the evidence.
