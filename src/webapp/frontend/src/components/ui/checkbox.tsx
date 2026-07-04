import React from "react";

interface CheckboxProps extends React.InputHTMLAttributes<HTMLInputElement> {
  indeterminate?: boolean;
}

export const Checkbox = React.forwardRef<HTMLInputElement, CheckboxProps>(
  ({ indeterminate, className = "", ...props }, ref) => {
    const defaultRef = React.useRef<HTMLInputElement>(null);
    const resolvedRef = (ref || defaultRef) as React.MutableRefObject<HTMLInputElement | null>;

    React.useEffect(() => {
      if (resolvedRef.current) {
        resolvedRef.current.indeterminate = !!indeterminate;
      }
    }, [resolvedRef, indeterminate]);

    return (
      <input
        type="checkbox"
        ref={resolvedRef}
        className={`h-4 w-4 rounded border-zinc-300 text-zinc-950 focus:ring-zinc-950 accent-zinc-900 cursor-pointer ${className}`}
        {...props}
      />
    );
  }
);

Checkbox.displayName = "Checkbox";
