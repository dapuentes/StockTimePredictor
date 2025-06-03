/**
 * Utility functions for parsing Python string literals to JavaScript objects
 */

/**
 * Parses a Python dictionary string literal to a JavaScript object
 * @param {string} pythonStr - String representation of a Python dictionary/list
 * @returns {Object|Array|string} - Parsed JavaScript object, array, or original string if parsing fails
 */
function parsePythonStringLiteral(pythonStr) {
    if (!pythonStr || typeof pythonStr !== 'string') {
        return pythonStr;
    }

    // If it's already a JavaScript object, return it
    if (typeof pythonStr === 'object') {
        return pythonStr;
    }

    try {
        // Clean up Python-specific syntax and convert to valid JSON
        let jsonStr = pythonStr
            // Convert Python booleans to JSON booleans
            .replace(/\bTrue\b/g, 'true')
            .replace(/\bFalse\b/g, 'false')
            .replace(/\bNone\b/g, 'null')
            // Convert single quotes to double quotes (basic conversion)
            .replace(/'/g, '"')
            // Handle Python tuple syntax (convert to arrays)
            .replace(/\(/g, '[')
            .replace(/\)/g, ']');

        // Try to parse as JSON
        return JSON.parse(jsonStr);
    } catch (error) {
        console.warn('Failed to parse Python string literal:', pythonStr, error);
        return pythonStr; // Return original string if parsing fails
    }
}

/**
 * Parses metadata object containing Python string literals
 * @param {Object} metadata - Metadata object that may contain Python string literals
 * @returns {Object} - Metadata object with parsed Python literals
 */
function parseMetadata(metadata) {
    if (!metadata || typeof metadata !== 'object') {
        return metadata;
    }

    const parsedMetadata = { ...metadata };

    // Parse common fields that are known to contain Python string literals
    const fieldsToParseAsObjects = ['best_params', 'metrics'];
    const fieldsToParseAsArrays = ['feature_importances', 'features_names'];

    fieldsToParseAsObjects.forEach(field => {
        if (parsedMetadata[field] && typeof parsedMetadata[field] === 'string') {
            parsedMetadata[field] = parsePythonStringLiteral(parsedMetadata[field]);
        }
    });

    fieldsToParseAsArrays.forEach(field => {
        if (parsedMetadata[field] && typeof parsedMetadata[field] === 'string') {
            const parsed = parsePythonStringLiteral(parsedMetadata[field]);
            // Ensure it's an array
            if (Array.isArray(parsed)) {
                parsedMetadata[field] = parsed;
            }
        }
    });

    return parsedMetadata;
}

/**
 * Enhanced format metadata function that also parses Python string literals
 * @param {Object} metadata - Raw metadata object
 * @returns {Object} - Formatted and parsed metadata object
 */
function formatMetadata(metadata) {
    if (!metadata) return null;
    
    try {
        let parsedMetadata = metadata;
        
        // If it's a string, try to parse it as JSON first
        if (typeof metadata === 'string') {
            try {
                parsedMetadata = JSON.parse(metadata);
            } catch (e) {
                // If regular JSON parsing fails, return the string as-is
                return metadata;
            }
        }
        
        // Now parse any Python string literals within the metadata
        return parseMetadata(parsedMetadata);
    } catch (e) {
        console.warn('Error formatting metadata:', e);
        return metadata;
    }
}

// Export all functions
export {
    parsePythonStringLiteral,
    parseMetadata,
    formatMetadata
};
