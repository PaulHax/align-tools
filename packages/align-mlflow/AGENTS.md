# UI integration

- Reuse existing MLflow UI components, state models, and query helpers before creating package-specific controls. Inspect the equivalent native view first.
- Reusing a table includes its relevant controls and behavior. For example, use the native trace filter builder and filter model instead of a separate popover with a fixed subset of fields.
- Chart and matrix selections should populate visible, editable filters beside the affected table. Prefer the actual field constraints over opaque labels such as "Matrix cell".
- Add a custom component only when the existing components cannot support the required behavior. Keep the custom part small and compose it with the native components.
