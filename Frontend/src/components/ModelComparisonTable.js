import React from 'react';
import { Table, Tag } from 'antd';
import dayjs from 'dayjs';

function ModelComparisonTable({ results }) {
     // Convertir el objeto results a un array para la tabla
     const dataSource = Object.values(results).sort((a, b) => dayjs(b.timestamp).unix() - dayjs(a.timestamp).unix()); // Ordenar por más reciente

    const columns = [
        { title: 'Modelo', dataIndex: 'modelType', key: 'modelType' },
        { title: 'Ticker', dataIndex: 'ticker', key: 'ticker' },
        { title: 'Rango Fechas', dataIndex: 'dateRange', key: 'dateRange' },        {
            title: 'MAPE',
            dataIndex: ['metrics', 'MAPE'], // Acceder a métrica anidada
            key: 'MAPE',
            align: 'right',
            render: (mape) => mape ? <Tag color={mape < 0.05 ? 'success' : mape < 0.10 ? 'processing' : mape < 0.15 ? 'warning' : 'error'}>{(mape * 100).toFixed(2)}%</Tag> : 'N/A',
             sorter: (a, b) => (a.metrics?.MAPE || 999) - (b.metrics?.MAPE || 999), // Ordenar por MAPE (menor es mejor)
         },{ title: 'RMSE', dataIndex: ['metrics', 'RMSE'], key: 'RMSE', align: 'right', render: (rmse) => rmse ? rmse.toFixed(4) : 'N/A', sorter: (a, b) => (a.metrics?.RMSE || 999) - (b.metrics?.RMSE || 999) },
        { title: 'MAE', dataIndex: ['metrics', 'MAE'], key: 'MAE', align: 'right', render: (mae) => mae ? mae.toFixed(4) : 'N/A', sorter: (a, b) => (a.metrics?.MAE || 999) - (b.metrics?.MAE || 999) },
        { title: 'Entrenado', dataIndex: 'timestamp', key: 'timestamp', render: (ts) => dayjs(ts).format('YYYY-MM-DD HH:mm'), sorter: (a,b) => dayjs(a.timestamp).unix() - dayjs(b.timestamp).unix() },
        { 
            title: 'Estado', 
            key: 'status', 
            render: () => <Tag color="success">Activo</Tag> // Todos los modelos mostrados están activos
        },
    ];

    return (
         <Table
            columns={columns}
            dataSource={dataSource}
            rowKey="id" // Usar el ID único como key
            size="middle"
            pagination={{ pageSize: 5 }}
        />
    );
}

export default ModelComparisonTable;