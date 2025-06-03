import React from 'react';
import { Card, Tag, Progress, Typography, List, Button, Space, Tooltip, Badge } from 'antd';
import { CloseOutlined, ReloadOutlined, ClockCircleOutlined } from '@ant-design/icons';
import dayjs from 'dayjs';
import relativeTime from 'dayjs/plugin/relativeTime';
import duration from 'dayjs/plugin/duration';

dayjs.extend(relativeTime);
dayjs.extend(duration);

const { Text, Title } = Typography;

const ActiveTrainingJobs = ({ 
    activeJobs, 
    onCancelJob, 
    onRetryJob,
    style 
}) => {
    if (!activeJobs || activeJobs.length === 0) {
        return null;
    }

    const getStatusColor = (status) => {
        switch (status) {
            case 'queued': return 'orange';
            case 'running': return 'blue';
            case 'submitting': return 'cyan';
            case 'completed': return 'green';
            case 'failed': return 'red';
            case 'timeout': return 'volcano';
            default: return 'default';
        }
    };

    const getStatusIcon = (status) => {
        switch (status) {
            case 'queued': return '⏳';
            case 'running': return '🔄';
            case 'submitting': return '📤';
            case 'completed': return '✅';
            case 'failed': return '❌';
            case 'timeout': return '⏰';
            default: return '❓';
        }
    };    const getElapsedTime = (startTime) => {
        const elapsed = dayjs().diff(dayjs(startTime));
        const duration = dayjs.duration(elapsed);
        
        if (duration.asMinutes() < 1) {
            return `${Math.floor(duration.asSeconds())}s`;
        } else if (duration.asHours() < 1) {
            return `${Math.floor(duration.asMinutes())}m ${Math.floor(duration.seconds())}s`;
        } else {
            return `${Math.floor(duration.asHours())}h ${Math.floor(duration.minutes())}m`;
        }
    };

    const getProgressBar = (job) => {
        if (job.progress && !isNaN(job.progress)) {
            return (
                <Progress 
                    percent={Math.round(job.progress)} 
                    size="small" 
                    status={job.status === 'failed' ? 'exception' : 'active'}
                    style={{ marginTop: '4px' }}
                />
            );
        }
        return null;
    };

    return (        <Card 
            title={
                <Space>
                    <Badge count={activeJobs.length} offset={[10, 0]}>
                        <span>🚀 Entrenamientos Activos</span>
                    </Badge>
                </Space>
            }
            size="small"
            style={style}
            bodyStyle={{ padding: '12px' }}
            extra={
                <Tooltip title="Los entrenamientos se ejecutan en segundo plano. Puedes continuar usando la aplicación normalmente.">
                    <ClockCircleOutlined style={{ color: '#1890ff' }} />
                </Tooltip>
            }
        >
            <List
                size="small"
                dataSource={activeJobs}
                renderItem={(job) => (
                    <List.Item
                        key={job.key}
                        actions={[
                            job.status === 'failed' && (
                                <Tooltip title="Reintentar entrenamiento">
                                    <Button
                                        type="text"
                                        size="small"
                                        icon={<ReloadOutlined />}
                                        onClick={() => onRetryJob && onRetryJob(job)}
                                    />
                                </Tooltip>
                            ),
                            <Tooltip title="Cancelar/Eliminar de la lista">
                                <Button
                                    type="text"
                                    size="small"
                                    danger
                                    icon={<CloseOutlined />}
                                    onClick={() => onCancelJob && onCancelJob(job)}
                                />
                            </Tooltip>
                        ].filter(Boolean)}
                    >
                        <List.Item.Meta
                            avatar={<span style={{ fontSize: '16px' }}>{getStatusIcon(job.status)}</span>}
                            title={
                                <Space>
                                    <Text strong>{job.ticker}</Text>
                                    <Tag color="default" size="small">{job.modelType.toUpperCase()}</Tag>
                                    <Tag color={getStatusColor(job.status)} size="small">
                                        {job.status.toUpperCase()}
                                    </Tag>
                                </Space>
                            }                            description={
                                <div>
                                    <div style={{ marginBottom: '4px' }}>
                                        <Text type="secondary" style={{ fontSize: '12px' }}>
                                            {job.message || 'Procesando...'}
                                        </Text>
                                    </div>
                                    
                                    {/* Progress bar if available */}
                                    {getProgressBar(job)}
                                    
                                    <div style={{ marginTop: '4px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <Text type="secondary" style={{ fontSize: '11px' }}>
                                            Duración: {getElapsedTime(job.startTime)}
                                        </Text>
                                        {job.jobId && (
                                            <Text type="secondary" style={{ fontSize: '10px' }}>
                                                ID: {job.jobId.slice(0, 8)}...
                                            </Text>
                                        )}
                                    </div>
                                </div>
                            }
                        />
                    </List.Item>
                )}
            />
        </Card>
    );
};

export default ActiveTrainingJobs;
