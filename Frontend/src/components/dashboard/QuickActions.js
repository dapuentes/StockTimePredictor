/**
 * QuickActions Component - Acciones rápidas para el usuario
 */
import React from 'react';
import { Button, Space, Dropdown, Tooltip, Badge } from 'antd';
import {
  ThunderboltOutlined,
  LineChartOutlined,
  DownloadOutlined,
  ReloadOutlined,
  SettingOutlined,
  PlusOutlined
} from '@ant-design/icons';

function QuickActions({
  onForecast,
  onTrain,
  onExport,
  onRefreshModels,
  isTraining,
  isForecastPending,
  selectedTicker,
  selectedModelType,
  hasForecastData
}) {
  const trainMenuItems = [
    {
      key: 'rf',
      label: 'Random Forest',
      icon: '🌲',
      onClick: () => onTrain('rf')
    },
    {
      key: 'lstm',
      label: 'LSTM Neural Network',
      icon: '🧠',
      onClick: () => onTrain('lstm')
    },
    {
      key: 'xgboost',
      label: 'XGBoost',
      icon: '🚀',
      onClick: () => onTrain('xgboost')
    },
    {
      key: 'prophet',
      label: 'Prophet',
      icon: '📊',
      onClick: () => onTrain('prophet')
    }
  ];

  return (
    <div style={{ 
      display: 'flex', 
      justifyContent: 'space-between', 
      alignItems: 'center',
      marginBottom: '24px',
      padding: '16px 24px',
      background: 'linear-gradient(135deg, #f8fafc 0%, #edf2f7 100%)',
      borderRadius: '12px',
      border: '1px solid #e2e8f0'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
        <ThunderboltOutlined style={{ color: '#4299e1', fontSize: '20px' }} />
        <span style={{ fontWeight: '600', color: '#2d3748' }}>
          Acciones Rápidas
        </span>
        {selectedTicker && (
          <Badge 
            count={selectedTicker}
            style={{ 
              backgroundColor: '#1a365d',
              marginLeft: '8px'
            }}
          />
        )}
      </div>
      
      <Space size="middle">
        {/* Botón principal de pronóstico */}
        <Tooltip title="Genera un pronóstico con el modelo seleccionado">
          <Button
            type="primary"
            size="large"
            icon={<LineChartOutlined />}
            onClick={onForecast}
            loading={isForecastPending}
            disabled={!selectedTicker || isForecastPending}
            style={{
              background: 'linear-gradient(135deg, #1a365d 0%, #2c5282 100%)',
              borderColor: 'transparent',
              boxShadow: '0 4px 12px rgba(26, 54, 93, 0.3)',
              height: '44px',
              paddingLeft: '24px',
              paddingRight: '24px',
              fontWeight: '600'
            }}
          >
            {isForecastPending ? 'Generando...' : 'Generar Pronóstico'}
          </Button>
        </Tooltip>

        {/* Dropdown de entrenamiento */}
        <Dropdown
          menu={{ items: trainMenuItems }}
          trigger={['click']}
          disabled={!selectedTicker || isTraining}
        >
          <Tooltip title="Entrenar un nuevo modelo">
            <Button
              size="large"
              icon={<PlusOutlined />}
              loading={isTraining}
              style={{
                height: '44px',
                borderColor: '#38a169',
                color: '#38a169'
              }}
            >
              Entrenar Modelo
            </Button>
          </Tooltip>
        </Dropdown>

        {/* Exportar */}
        <Tooltip title="Exportar pronóstico a CSV">
          <Button
            icon={<DownloadOutlined />}
            onClick={onExport}
            disabled={!hasForecastData}
            style={{ height: '44px' }}
          >
            Exportar
          </Button>
        </Tooltip>

        {/* Refrescar modelos */}
        <Tooltip title="Actualizar lista de modelos">
          <Button
            icon={<ReloadOutlined />}
            onClick={onRefreshModels}
            style={{ height: '44px' }}
          />
        </Tooltip>
      </Space>
    </div>
  );
}

export default QuickActions;
