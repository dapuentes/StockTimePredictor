/**
 * Header Component - Encabezado moderno y profesional
 */
import React from 'react';
import { Switch, Button, Space, Badge, Tooltip } from 'antd';
import { 
  SunOutlined, 
  MoonOutlined, 
  QuestionCircleOutlined,
  GithubOutlined,
  BellOutlined
} from '@ant-design/icons';

function AppHeader({ 
  currentTheme, 
  onThemeToggle, 
  onHelpClick,
  activeJobsCount = 0 
}) {
  return (
    <header className="app-header">
      <div className="app-header__logo">
        <div className="app-header__logo-icon">
          📈
        </div>
        <div>
          <h1 className="app-header__title">StockTime Predictor</h1>
          <div className="app-header__subtitle">
            Predicción inteligente de series financieras
          </div>
        </div>
      </div>
      
      <div className="app-header__actions">
        {/* Indicador de trabajos activos */}
        {activeJobsCount > 0 && (
          <Tooltip title={`${activeJobsCount} entrenamiento${activeJobsCount > 1 ? 's' : ''} en progreso`}>
            <Badge count={activeJobsCount} offset={[-5, 5]}>
              <Button 
                type="text" 
                icon={<BellOutlined style={{ color: 'rgba(255,255,255,0.85)', fontSize: '16px' }} />}
              />
            </Badge>
          </Tooltip>
        )}
        
        {/* Enlace a GitHub */}
        <Tooltip title="Ver en GitHub">
          <Button
            type="text"
            icon={<GithubOutlined style={{ color: 'rgba(255,255,255,0.85)', fontSize: '16px' }} />}
            href="https://github.com/dapuentes/StockTimePredictor"
            target="_blank"
          />
        </Tooltip>
        
        {/* Botón de ayuda */}
        <Button
          type="text"
          icon={<QuestionCircleOutlined style={{ color: 'rgba(255,255,255,0.85)', fontSize: '16px' }} />}
          onClick={onHelpClick}
        >
          <span style={{ color: 'rgba(255,255,255,0.85)' }}>Ayuda</span>
        </Button>
        
        {/* Switch de tema */}
        <Space>
          <Switch
            checkedChildren={<MoonOutlined />}
            unCheckedChildren={<SunOutlined />}
            onChange={onThemeToggle}
            checked={currentTheme === 'dark'}
          />
        </Space>
      </div>
    </header>
  );
}

export default AppHeader;
