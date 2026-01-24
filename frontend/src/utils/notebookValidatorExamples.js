// Example usage of notebook URL validation
// Import in your component:
// import { validateNotebookUrl, testNotebookUrl } from '../utils/notebookValidator';

// Example 1: Detailed validation with error handling
const checkNotebookDetailed = async (url) => {
  const result = await validateNotebookUrl(url, 5000); // 5 second timeout
  
  if (result.valid) {
    console.log(`✓ Valid notebook with ${result.details.cellCount} cells`);
    return true;
  } else {
    console.error(`✗ Invalid notebook: ${result.error}`);
    
    switch (result.error) {
      case 'api_error':
        console.error('API Error:', result.details.detail);
        break;
      case 'timeout':
        console.error('Request timed out after', result.details.timeout, 'ms');
        break;
      case 'invalid_notebook_structure':
        console.error('Notebook validation failed:', result.details.problems);
        break;
      case 'network_error':
        console.error('Network error:', result.details.message);
        break;
      default:
        console.error('Details:', result.details);
    }
    
    return false;
  }
};

// Example 2: Simple boolean check
const checkNotebookSimple = async (url) => {
  const isValid = await testNotebookUrl(url);
  console.log(`Notebook valid: ${isValid}`);
  return isValid;
};

// Example 3: React component usage
const NotebookUrlChecker = ({ url }) => {
  const [validation, setValidation] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const handleCheck = async () => {
    setLoading(true);
    const result = await validateNotebookUrl(url);
    setValidation(result);
    setLoading(false);
  };
  
  return (
    <div>
      <button onClick={handleCheck} disabled={loading}>
        {loading ? 'Checking...' : 'Validate Notebook'}
      </button>
      
      {validation && (
        <div>
          {validation.valid ? (
            <Alert severity="success">
              Valid notebook with {validation.details.cellCount} cells
            </Alert>
          ) : (
            <Alert severity="error">
              Invalid notebook: {validation.error}
              {validation.details && (
                <pre>{JSON.stringify(validation.details, null, 2)}</pre>
              )}
            </Alert>
          )}
        </div>
      )}
    </div>
  );
};

// Example 4: Replace your current testUrl function with this:
const testUrl = async (url) => {
  try {
    const result = await validateNotebookUrl(url, 3000);
    
    if (result.valid) {
      console.log(`✓ Tested URL ${url}, valid notebook`);
      return true;
    } else {
      console.warn(`✗ Tested URL ${url}, invalid:`, result.error);
      
      // Handle specific errors
      if (result.error === 'api_error' && result.details?.detail === 'Not Found') {
        console.error('Notebook endpoint not found (404)');
      }
      
      return false;
    }
  } catch (error) {
    console.error('Unexpected error testing URL:', error);
    return false;
  }
};

export { checkNotebookDetailed, checkNotebookSimple, testUrl };
