// Simple test to verify the JSON generator functionality
import JSONGenerator from '../components/blockly/jsonGenerator';

// Mock Blockly workspace for testing
const mockWorkspace = {
  getTopBlocks: () => [],
};

const mockBlock = {
  id: 'test-block-1',
  getFieldValue: (field) => {
    switch(field) {
      case 'POWER': return 50;
      case 'CHANNEL': return 'LED';
      case 'SECONDS': return 2;
      default: return '';
    }
  },
  getNextBlock: () => null,
};

describe('JSONGenerator', () => {
  test('should generate proper JSON structure for empty workspace', () => {
    const result = JSONGenerator.workspaceToCode(mockWorkspace);
    const parsed = JSON.parse(result);
    
    expect(parsed).toHaveProperty('steps');
    expect(Array.isArray(parsed.steps)).toBe(true);
  });

  test('should generate correct JSON for set_laser_power_block', () => {
    const result = JSONGenerator['set_laser_power_block'](mockBlock);
    const parsed = JSON.parse(result.replace(/,$/, '')); // Remove trailing comma
    
    expect(parsed).toHaveProperty('id', 'test-block-1');
    expect(parsed).toHaveProperty('stepName', 'Set Laser Power');
    expect(parsed).toHaveProperty('mainFuncName', 'set_laser_power');
    expect(parsed.mainParams).toHaveProperty('power', 50);
    expect(parsed.mainParams).toHaveProperty('channel', 'LED');
  });

  test('should generate correct JSON for wait_time_block', () => {
    const result = JSONGenerator['wait_time_block'](mockBlock);
    const parsed = JSON.parse(result.replace(/,$/, '')); // Remove trailing comma
    
    expect(parsed).toHaveProperty('id', 'test-block-1');
    expect(parsed).toHaveProperty('stepName', 'Wait Time');
    expect(parsed).toHaveProperty('mainFuncName', 'wait_time');
    expect(parsed.mainParams).toHaveProperty('seconds', 2);
  });
});