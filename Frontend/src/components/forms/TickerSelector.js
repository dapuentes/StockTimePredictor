/**
 * TickerSelector Component - Selector moderno de acciones
 * Con búsqueda, categorías y favoritos
 */
import React, { useState, useMemo } from 'react';
import { Select, Tag, Input, Space, Divider, Typography } from 'antd';
import { 
  SearchOutlined, 
  StarOutlined, 
  StarFilled,
  GlobalOutlined,
  BankOutlined,
  ShopOutlined,
  ThunderboltOutlined,
  DollarOutlined
} from '@ant-design/icons';

const { Text } = Typography;

// Categorías de tickers
const TICKER_CATEGORIES = {
  tech: {
    name: 'Tecnología',
    icon: <ThunderboltOutlined />,
    color: '#4299e1',
    tickers: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ORCL', 'CRM', 'ADBE', 'INTC', 'AMD']
  },
  financial: {
    name: 'Finanzas',
    icon: <BankOutlined />,
    color: '#38a169',
    tickers: ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B', 'V', 'MA', 'AXP']
  },
  healthcare: {
    name: 'Salud',
    icon: '🏥',
    color: '#ed8936',
    tickers: ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'LLY']
  },
  retail: {
    name: 'Consumo',
    icon: <ShopOutlined />,
    color: '#e53e3e',
    tickers: ['WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'NKE', 'SBUX', 'DIS', 'COST']
  },
  latam: {
    name: 'Latinoamérica',
    icon: <GlobalOutlined />,
    color: '#805ad5',
    tickers: ['NU', 'VALE', 'ITUB', 'BBD', 'PBR', 'GGAL', 'YPF', 'MELI', 'GLOB']
  },
  etf: {
    name: 'ETFs',
    icon: <DollarOutlined />,
    color: '#319795',
    tickers: ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'VTI', 'VEA', 'VWO']
  },
  crypto: {
    name: 'Cripto',
    icon: '₿',
    color: '#d69e2e',
    tickers: ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'SOL-USD', 'MATIC-USD']
  }
};

function TickerSelector({ 
  value, 
  onChange, 
  availableTickers,
  style 
}) {
  const [favorites, setFavorites] = useState(() => {
    const saved = localStorage.getItem('favorite_tickers');
    return saved ? JSON.parse(saved) : ['NU', 'AAPL', 'GOOGL'];
  });
  const [searchText, setSearchText] = useState('');

  const toggleFavorite = (ticker, e) => {
    e.stopPropagation();
    const newFavorites = favorites.includes(ticker)
      ? favorites.filter(t => t !== ticker)
      : [...favorites, ticker];
    setFavorites(newFavorites);
    localStorage.setItem('favorite_tickers', JSON.stringify(newFavorites));
  };

  // Obtener categoría de un ticker
  const getTickerCategory = (ticker) => {
    for (const [key, category] of Object.entries(TICKER_CATEGORIES)) {
      if (category.tickers.includes(ticker)) {
        return { key, ...category };
      }
    }
    return null;
  };

  // Opciones filtradas y ordenadas
  const options = useMemo(() => {
    const filtered = availableTickers.filter(ticker => 
      ticker.toLowerCase().includes(searchText.toLowerCase())
    );

    // Ordenar: favoritos primero, luego alfabéticamente
    return filtered.sort((a, b) => {
      const aFav = favorites.includes(a);
      const bFav = favorites.includes(b);
      if (aFav && !bFav) return -1;
      if (!aFav && bFav) return 1;
      return a.localeCompare(b);
    }).map(ticker => {
      const category = getTickerCategory(ticker);
      const isFavorite = favorites.includes(ticker);
      
      return {
        value: ticker,
        label: (
          <div style={{ 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'space-between',
            width: '100%'
          }}>
            <Space>
              <span style={{ fontWeight: '600', minWidth: '70px' }}>{ticker}</span>
              {category && (
                <Tag 
                  color={category.color} 
                  style={{ fontSize: '10px', margin: 0 }}
                >
                  {category.name}
                </Tag>
              )}
            </Space>
            <span 
              onClick={(e) => toggleFavorite(ticker, e)}
              style={{ cursor: 'pointer' }}
            >
              {isFavorite ? (
                <StarFilled style={{ color: '#fbbf24' }} />
              ) : (
                <StarOutlined style={{ color: '#cbd5e0' }} />
              )}
            </span>
          </div>
        )
      };
    });
  }, [availableTickers, searchText, favorites]);

  // Dropdown personalizado
  const dropdownRender = (menu) => (
    <>
      <div style={{ padding: '8px 12px' }}>
        <Input
          placeholder="Buscar ticker..."
          prefix={<SearchOutlined style={{ color: '#a0aec0' }} />}
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
          allowClear
          style={{ borderRadius: '8px' }}
        />
      </div>
      
      {/* Favoritos rápidos */}
      {favorites.length > 0 && !searchText && (
        <div style={{ padding: '8px 12px' }}>
          <Text type="secondary" style={{ fontSize: '11px', textTransform: 'uppercase' }}>
            Favoritos
          </Text>
          <div style={{ marginTop: '4px', display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
            {favorites.slice(0, 6).map(ticker => (
              <Tag
                key={ticker}
                color={value === ticker ? 'blue' : 'default'}
                style={{ cursor: 'pointer', margin: 0 }}
                onClick={() => onChange(ticker)}
              >
                <StarFilled style={{ color: '#fbbf24', marginRight: '4px', fontSize: '10px' }} />
                {ticker}
              </Tag>
            ))}
          </div>
        </div>
      )}
      
      <Divider style={{ margin: '8px 0' }} />
      
      {menu}
    </>
  );

  return (
    <Select
      value={value}
      onChange={onChange}
      options={options}
      style={{ width: '100%', ...style }}
      placeholder="Selecciona una acción"
      showSearch={false} // Usamos nuestro propio buscador
      dropdownRender={dropdownRender}
      listHeight={300}
      optionLabelProp="value"
      dropdownStyle={{ 
        borderRadius: '12px',
        boxShadow: '0 10px 40px rgba(0,0,0,0.15)'
      }}
    />
  );
}

export default TickerSelector;
