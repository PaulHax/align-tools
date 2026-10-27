# UI integration

- Add a custom component only when the existing components cannot support the required behavior. Keep the custom part small and compose it with the native components.
- Reuse shared renderers, interaction logic, and state wiring across single-session and comparison views so equivalent actions behave consistently.
- Fix behavior in the shared path and remove superseded rendering or interaction infrastructure when consolidating views.
