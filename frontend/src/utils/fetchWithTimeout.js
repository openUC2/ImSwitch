/**
 * Cross-browser compatible fetch with timeout
 *
 * Uses AbortController with setTimeout for maximum browser compatibility
 * instead of AbortSignal.timeout() which has limited support.
 *
 * @param {string} url - The URL to fetch
 * @param {Object} options - Fetch options (method, headers, body, etc.)
 * @param {number} timeout - Timeout in milliseconds (default: 5000)
 * @returns {Promise<Response>} - Fetch response
 *
 * @example
 * // Basic usage
 * const response = await fetchWithTimeout('/api/data', { method: 'GET' });
 *
 * // Custom timeout
 * const response = await fetchWithTimeout('/api/slow', { method: 'POST' }, 10000);
 *
 * // With error handling
 * try {
 *   const response = await fetchWithTimeout('/api/version', { method: 'GET' }, 3000);
 *   if (response.ok) {
 *     const data = await response.json();
 *     console.log(data);
 *   }
 * } catch (error) {
 *   if (error.name === 'AbortError') {
 *     console.log('Request timed out');
 *   } else {
 *     console.error('Request failed:', error);
 *   }
 * }
 */
export const fetchWithTimeout = async (url, options = {}, timeout = 5000) => {
  // Create abort controller for timeout handling
  const controller = new AbortController();

  // Set up timeout
  const timeoutId = setTimeout(() => {
    controller.abort();
  }, timeout);

  try {
    // Merge abort signal with provided options
    const fetchOptions = {
      ...options,
      signal: controller.signal,
    };

    // Perform the fetch request
    const response = await fetch(url, fetchOptions);

    // Clear timeout on successful response
    clearTimeout(timeoutId);

    return response;
  } catch (error) {
    // Clear timeout on error
    clearTimeout(timeoutId);

    // Re-throw the error for handling by caller
    throw error;
  }
};

/**
 * Convenience wrapper for GET requests with timeout
 *
 * @param {string} url - The URL to fetch
 * @param {number} timeout - Timeout in milliseconds (default: 5000)
 * @returns {Promise<Response>} - Fetch response
 */
export const fetchGetWithTimeout = (url, timeout = 5000) => {
  return fetchWithTimeout(url, { method: "GET" }, timeout);
};

/**
 * Convenience wrapper for POST requests with timeout
 *
 * @param {string} url - The URL to fetch
 * @param {Object} data - Data to send in request body
 * @param {number} timeout - Timeout in milliseconds (default: 5000)
 * @returns {Promise<Response>} - Fetch response
 */
export const fetchPostWithTimeout = (url, data, timeout = 5000) => {
  return fetchWithTimeout(
    url,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    },
    timeout
  );
};

/**
 * Check if an error is a timeout/abort error
 *
 * @param {Error} error - The error to check
 * @returns {boolean} - True if error is timeout related
 */
export const isTimeoutError = (error) => {
  return error.name === "AbortError";
};

export default fetchWithTimeout;
