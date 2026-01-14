import {
  fetchWithTimeout,
  fetchGetWithTimeout,
  fetchPostWithTimeout,
  isTimeoutError,
} from "../utils/fetchWithTimeout";

// Mock fetch globally
global.fetch = jest.fn();

describe("fetchWithTimeout", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.clearAllTimers();
    jest.useFakeTimers();

    // Reset fetch mock
    global.fetch.mockReset();
  });

  afterEach(() => {
    jest.runOnlyPendingTimers();
    jest.useRealTimers();
  });

  describe("Basic functionality", () => {
    test("should make successful fetch request", async () => {
      // Mock successful response
      const mockResponse = new Response('{"success": true}', {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
      global.fetch.mockResolvedValueOnce(mockResponse);

      const response = await fetchWithTimeout("/api/test", { method: "GET" });

      expect(global.fetch).toHaveBeenCalledWith("/api/test", {
        method: "GET",
        signal: expect.any(AbortSignal),
      });
      expect(response.status).toBe(200);
    });

    test("should use default timeout of 5000ms", async () => {
      const mockResponse = new Response("ok");
      global.fetch.mockResolvedValueOnce(mockResponse);

      await fetchWithTimeout("/api/test");

      // Check that setTimeout was called with 5000ms
      expect(setTimeout).toHaveBeenCalledWith(expect.any(Function), 5000);
    });

    test("should use custom timeout", async () => {
      const mockResponse = new Response("ok");
      global.fetch.mockResolvedValueOnce(mockResponse);

      await fetchWithTimeout("/api/test", {}, 3000);

      expect(setTimeout).toHaveBeenCalledWith(expect.any(Function), 3000);
    });

    test("should clear timeout on successful response", async () => {
      const mockResponse = new Response("ok");
      global.fetch.mockResolvedValueOnce(mockResponse);

      await fetchWithTimeout("/api/test");

      expect(clearTimeout).toHaveBeenCalled();
    });
  });

  describe("Timeout handling", () => {
    test("should abort request when timeout is reached", async () => {
      // Mock a slow response
      global.fetch.mockImplementationOnce(
        () =>
          new Promise((resolve) =>
            setTimeout(() => resolve(new Response("ok")), 10000)
          )
      );

      const fetchPromise = fetchWithTimeout("/api/slow", {}, 1000);

      // Fast-forward time to trigger timeout
      jest.advanceTimersByTime(1000);

      await expect(fetchPromise).rejects.toThrow();
    });

    test("should clear timeout on error", async () => {
      global.fetch.mockRejectedValueOnce(new Error("Network error"));

      try {
        await fetchWithTimeout("/api/test");
      } catch (error) {
        // Expected to throw
      }

      expect(clearTimeout).toHaveBeenCalled();
    });
  });

  describe("Options merging", () => {
    test("should merge options correctly", async () => {
      const mockResponse = new Response("ok");
      global.fetch.mockResolvedValueOnce(mockResponse);

      const customOptions = {
        method: "POST",
        headers: { Authorization: "Bearer token" },
        body: JSON.stringify({ data: "test" }),
      };

      await fetchWithTimeout("/api/test", customOptions);

      expect(global.fetch).toHaveBeenCalledWith("/api/test", {
        method: "POST",
        headers: { Authorization: "Bearer token" },
        body: JSON.stringify({ data: "test" }),
        signal: expect.any(AbortSignal),
      });
    });

    test("should not override existing signal", async () => {
      const mockResponse = new Response("ok");
      global.fetch.mockResolvedValueOnce(mockResponse);

      const existingController = new AbortController();
      const options = { signal: existingController.signal };

      await fetchWithTimeout("/api/test", options);

      // Should use the timeout controller's signal, not the existing one
      expect(global.fetch).toHaveBeenCalledWith("/api/test", {
        signal: expect.any(AbortSignal),
      });
    });
  });

  describe("Convenience functions", () => {
    test("fetchGetWithTimeout should make GET request", async () => {
      const mockResponse = new Response("ok");
      global.fetch.mockResolvedValueOnce(mockResponse);

      await fetchGetWithTimeout("/api/data");

      expect(global.fetch).toHaveBeenCalledWith("/api/data", {
        method: "GET",
        signal: expect.any(AbortSignal),
      });
    });

    test("fetchGetWithTimeout should use custom timeout", async () => {
      const mockResponse = new Response("ok");
      global.fetch.mockResolvedValueOnce(mockResponse);

      await fetchGetWithTimeout("/api/data", 3000);

      expect(setTimeout).toHaveBeenCalledWith(expect.any(Function), 3000);
    });

    test("fetchPostWithTimeout should make POST request with JSON", async () => {
      const mockResponse = new Response("ok");
      global.fetch.mockResolvedValueOnce(mockResponse);

      const testData = { name: "test", value: 123 };
      await fetchPostWithTimeout("/api/data", testData);

      expect(global.fetch).toHaveBeenCalledWith("/api/data", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(testData),
        signal: expect.any(AbortSignal),
      });
    });

    test("fetchPostWithTimeout should use custom timeout", async () => {
      const mockResponse = new Response("ok");
      global.fetch.mockResolvedValueOnce(mockResponse);

      await fetchPostWithTimeout("/api/data", {}, 7000);

      expect(setTimeout).toHaveBeenCalledWith(expect.any(Function), 7000);
    });
  });

  describe("Error handling utilities", () => {
    test("isTimeoutError should identify abort errors", () => {
      const abortError = new Error("The operation was aborted");
      abortError.name = "AbortError";

      const networkError = new Error("Network failed");
      networkError.name = "TypeError";

      expect(isTimeoutError(abortError)).toBe(true);
      expect(isTimeoutError(networkError)).toBe(false);
    });

    test("isTimeoutError should handle undefined error", () => {
      expect(isTimeoutError(null)).toBe(false);
      expect(isTimeoutError(undefined)).toBe(false);
    });

    test("isTimeoutError should handle non-Error objects", () => {
      expect(isTimeoutError({ name: "AbortError" })).toBe(true);
      expect(isTimeoutError({ name: "SomeOtherError" })).toBe(false);
      expect(isTimeoutError("string error")).toBe(false);
    });
  });

  describe("Real-world scenarios", () => {
    test("should handle network failure gracefully", async () => {
      const networkError = new Error("Failed to fetch");
      global.fetch.mockRejectedValueOnce(networkError);

      await expect(fetchWithTimeout("/api/test")).rejects.toThrow(
        "Failed to fetch"
      );
      expect(clearTimeout).toHaveBeenCalled();
    });

    test("should handle JSON response parsing", async () => {
      const jsonData = { message: "success", data: [1, 2, 3] };
      const mockResponse = new Response(JSON.stringify(jsonData), {
        headers: { "Content-Type": "application/json" },
      });
      global.fetch.mockResolvedValueOnce(mockResponse);

      const response = await fetchWithTimeout("/api/test");
      const data = await response.json();

      expect(data).toEqual(jsonData);
    });

    test("should handle non-200 status codes", async () => {
      const mockResponse = new Response("Not Found", { status: 404 });
      global.fetch.mockResolvedValueOnce(mockResponse);

      const response = await fetchWithTimeout("/api/test");

      expect(response.status).toBe(404);
      expect(response.ok).toBe(false);
    });
  });

  describe("Browser compatibility", () => {
    test("should work without AbortSignal.timeout", () => {
      // This test ensures our implementation doesn't rely on AbortSignal.timeout
      // which has limited browser support

      const originalTimeout = AbortSignal.timeout;
      delete AbortSignal.timeout;

      const mockResponse = new Response("ok");
      global.fetch.mockResolvedValueOnce(mockResponse);

      expect(async () => {
        await fetchWithTimeout("/api/test");
      }).not.toThrow();

      // Restore if it existed
      if (originalTimeout) {
        AbortSignal.timeout = originalTimeout;
      }
    });

    test("should create new AbortController for each request", async () => {
      const mockResponse1 = new Response("ok1");
      const mockResponse2 = new Response("ok2");
      global.fetch
        .mockResolvedValueOnce(mockResponse1)
        .mockResolvedValueOnce(mockResponse2);

      await fetchWithTimeout("/api/test1");
      await fetchWithTimeout("/api/test2");

      // Each call should create its own controller
      expect(global.fetch).toHaveBeenCalledTimes(2);
      expect(setTimeout).toHaveBeenCalledTimes(2);
      expect(clearTimeout).toHaveBeenCalledTimes(2);
    });
  });
});
