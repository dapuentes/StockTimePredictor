import React from 'react';
import '../App.css';

/**
 * A functional React component that displays an error message with an optional close button.
 *
 * @param {Object} props - The props object for the component.
 * @param {string} props.message - The error message to display.
 * @param {Function} [props.onClose] - An optional callback function to handle the close button click event.
 * @returns {JSX.Element} The rendered error message component.
 */
const ErrorMessage = ({ message, onClose }) => (
    <div className="error-message">
        <span>{message}</span>
        {onClose && <button onClick={onClose} className="close-error-btn">✖</button>}
    </div>
);

export default ErrorMessage;