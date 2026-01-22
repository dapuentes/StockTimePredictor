/**
 * ModelSelector Component - Selector de tipo de modelo con información visual
 */
import React from 'react';
import { Radio, Space, Tag, Tooltip, Badge } from 'antd';
import {
  ExperimentOutlined,
  RobotOutlined,
  ThunderboltOutlined,
  LineChartOutlined
} from '@ant-design/icons';

const MODEL_INFO = {
  rf: {
    name: 'Random Forest',
    shortName: 'RF',
    icon: '🌲',
    antIcon: <ExperimentOutlined />,
    color: '#38a169',
    description: 'Modelo de ensemble basado en árboles. Robusto y rápido de entrenar.',
    pros: ['Rápido', 'Interpretable', 'Robusto'],
    speed: 'Rápido',
    accuracy: 'Alta'
  },
  lstm: {
    name: 'LSTM Neural Network',
    shortName: 'LSTM',
    icon: '🧠',
    antIcon: <RobotOutlined />,
    color: '#805ad5',
    description: 'Red neuronal recurrente. Ideal para patrones temporales complejos.',
    pros: ['Patrones complejos', 'Memoria larga', 'Secuencial'],
    speed: 'Lento',
    accuracy: 'Muy alta'
  },
  xgboost: {
    name: 'XGBoost',
    shortName: 'XGB',
    icon: '🚀',
    antIcon: <ThunderboltOutlined />,
    color: '#ed8936',
    description: 'Gradient boosting optimizado. Excelente rendimiento general.',
    pros: ['Muy preciso', 'Eficiente', 'Flexible'],
    speed: 'Medio',
    accuracy: 'Muy alta'
  },
  prophet: {
    name: 'Prophet',
    shortName: 'Prophet',
    icon: '📊',
    antIcon: <LineChartOutlined />,
    color: '#4299e1',
    description: 'Modelo de Facebook para series temporales con estacionalidad.',
    pros: ['Estacionalidad', 'Tendencias', 'Feriados'],
    speed: 'Rápido',
    accuracy: 'Alta'
  }
};

function ModelSelector({
  value,
  onChange,
  availableModelTypes,
  availableModels,
  style
}) {
  const getModelCount = (modelType) => {
    return availableModels?.[modelType]?.total_models || 0;
  };

  return (
    <div style={style}>
      <Radio.Group
        value={value}
        onChange={(e) => onChange(e.target.value)}
        style={{ width: '100%' }}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          {availableModelTypes.map(modelType => {
            const model = MODEL_INFO[modelType];
            const modelCount = getModelCount(modelType);
            const isSelected = value === modelType;
            
            return (
              <Tooltip
                key={modelType}
                title={
                  <div style={{ maxWidth: '250px' }}>
                    <div style={{ fontWeight: 'bold', marginBottom: '8px' }}>
                      {model.icon} {model.name}
                    </div>
                    <div style={{ marginBottom: '8px', opacity: 0.9 }}>
                      {model.description}
                    </div>
                    <div style={{ display: 'flex', gap: '4px', flexWrap: 'wrap' }}>
                      {model.pros.map(pro => (
                        <Tag key={pro} color="blue" style={{ margin: 0, fontSize: '10px' }}>
                          {pro}
                        </Tag>
                      ))}
                    </div>
                    <div style={{ marginTop: '8px', fontSize: '11px' }}>
                      <span>⚡ Velocidad: {model.speed}</span>
                      <span style={{ marginLeft: '12px' }}>🎯 Precisión: {model.accuracy}</span>
                    </div>
                  </div>
                }
                placement="right"
              >
                <Radio.Button
                  value={modelType}
                  style={{
                    width: '100%',
                    height: 'auto',
                    padding: '12px 16px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    borderRadius: '8px',
                    marginBottom: '8px',
                    borderColor: isSelected ? model.color : '#e2e8f0',
                    backgroundColor: isSelected ? `${model.color}10` : 'white',
                    transition: 'all 0.2s ease'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <span style={{ 
                      fontSize: '20px',
                      width: '36px',
                      height: '36px',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      backgroundColor: isSelected ? `${model.color}20` : '#f7fafc',
                      borderRadius: '8px'
                    }}>
                      {model.icon}
                    </span>
                    <div>
                      <div style={{ 
                        fontWeight: '600', 
                        color: isSelected ? model.color : '#2d3748',
                        fontSize: '14px'
                      }}>
                        {model.name}
                      </div>
                      <div style={{ 
                        fontSize: '11px', 
                        color: '#718096'
                      }}>
                        {model.speed} • {model.accuracy}
                      </div>
                    </div>
                  </div>
                  
                  {modelCount > 0 && (
                    <Badge 
                      count={`${modelCount} modelo${modelCount > 1 ? 's' : ''}`}
                      style={{ 
                        backgroundColor: model.color,
                        fontSize: '10px'
                      }}
                    />
                  )}
                </Radio.Button>
              </Tooltip>
            );
          })}
        </Space>
      </Radio.Group>
    </div>
  );
}

export default ModelSelector;
