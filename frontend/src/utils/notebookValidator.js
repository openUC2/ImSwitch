// src/utils/notebookValidator.js
// Utility to validate if a URL returns a valid Jupyter notebook

/**
 * Validate if a URL returns a valid Jupyter notebook
 * 
 * Checks for:
 * - FastAPI error responses like {"detail":"Not Found"}
 * - Timeout errors
 * - Valid notebook JSON structure (cells array, metadata.language, etc.)
 * 
 * @param {string} url - URL to validate
 * @param {number} timeout - Request timeout in milliseconds (default: 3000)
 * @returns {Promise<{valid: boolean, error?: string, details?: object}>}
 */
export const validateNotebookUrl = async (url, timeout = 3000) => {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    const response = await fetch(url, {
      method: 'GET',
      signal: controller.signal,
      headers: {
        'Accept': 'application/json',
      },
    });
    
    clearTimeout(timeoutId);
    
    // Check HTTP status
    if (!response.ok) {
      // Try to parse JSON error response (FastAPI style)
      try {
        const errorData = await response.json();
        
        // FastAPI error: {"detail": "Not Found"}
        if (errorData && errorData.detail) {
          return {
            valid: false,
            error: 'api_error',
            details: errorData,
          };
        }
        
        return {
          valid: false,
          error: `http_${response.status}`,
          details: errorData,
        };
      } catch (parseError) {
        // Not JSON, return status error
        return {
          valid: false,
          error: `http_${response.status}`,
          details: { status: response.status, statusText: response.statusText },
        };
      }
    }
    
    // 200 OK - parse and validate notebook structure
    let data;
    try {
      data = await response.json();
    } catch (parseError) {
      return {
        valid: false,
        error: 'invalid_json',
        details: { message: 'Response is not valid JSON' },
      };
    }
    
    // Validate notebook structure
    const validation = validateNotebookStructure(data);
    
    if (!validation.valid) {
      return {
        valid: false,
        error: 'invalid_notebook_structure',
        details: validation.details,
      };
    }
    
    return {
      valid: true,
      details: {
        cellCount: data.cells?.length || 0,
      },
    };
    
  } catch (error) {
    // Timeout or network error
    if (error.name === 'AbortError') {
      return {
        valid: false,
        error: 'timeout',
        details: { timeout },
      };
    }
    
    return {
      valid: false,
      error: 'network_error',
      details: { message: error.message },
    };
  }
};

/**
 * Validate notebook JSON structure
 * 
 * @param {object} data - Parsed JSON data
 * @returns {object} - {valid: boolean, details?: object}
 */
const validateNotebookStructure = (data) => {
  // Must be an object
  if (!data || typeof data !== 'object' || Array.isArray(data)) {
    return {
      valid: false,
      details: { message: 'Top-level JSON is not an object' },
    };
  }
  
  // Must have cells array
  if (!Array.isArray(data.cells)) {
    return {
      valid: false,
      details: { message: 'Missing or invalid "cells" array' },
    };
  }
  
  // Validate each cell
  const problems = [];
  data.cells.forEach((cell, index) => {
    if (!cell || typeof cell !== 'object') {
      problems.push(`Cell ${index + 1}: not an object`);
      return;
    }
    
    const meta = cell.metadata;
    if (!meta || typeof meta !== 'object') {
      problems.push(`Cell ${index + 1}: missing metadata object`);
      return;
    }
    
    if (!meta.language) {
      problems.push(`Cell ${index + 1}: missing metadata.language`);
    }
    
    // Note: metadata.id is optional for new cells, so we don't enforce it
    
    if (!cell.cell_type) {
      problems.push(`Cell ${index + 1}: missing cell_type`);
    }
    
    if (!cell.source) {
      problems.push(`Cell ${index + 1}: missing source`);
    }
  });
  
  if (problems.length > 0) {
    return {
      valid: false,
      details: { problems },
    };
  }
  
  return { valid: true };
};

/**
 * Simple test function for quick validation (returns boolean only)
 * 
 * @param {string} url - URL to test
 * @param {number} timeout - Timeout in milliseconds
 * @returns {Promise<boolean>} - true if valid notebook, false otherwise
 */
export const testNotebookUrl = async (url, timeout = 3000) => {
  const result = await validateNotebookUrl(url, timeout);
  return result.valid;
};

export default validateNotebookUrl;
