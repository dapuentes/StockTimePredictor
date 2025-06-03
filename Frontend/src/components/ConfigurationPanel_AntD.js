import React, { useState, useEffect } from 'react';
import {
    Form,       
    Select,     
    DatePicker, 
    InputNumber,
    Button,  
    Divider,  
    Typography,
    Spin,
    Alert,
    Tooltip,
    Modal,
    Descriptions,
    Tag,
    Space,
    message
} from 'antd';
import dayjs from 'dayjs';
import { InfoCircleOutlined, EyeOutlined, ReloadOutlined, DeleteOutlined } from '@ant-design/icons';
import { formatMetadata } from '../utils/pythonUtils';

const { RangePicker } = DatePicker;
const { Option } = Select;
const { Title } = Typography;

// Convertir Date a dayjs y viceversa para el RangePicker
const dateToDayjs = (date) => (date ? dayjs(date) : null);
const dayjsToDate = (dayjsObj) => (dayjsObj ? dayjsObj.toDate() : null);

// Helper function to extract ticker from model name
const extractTickerFromModelName = (modelName) => {
    if (!modelName || typeof modelName !== 'string') return null;
    
    // Common patterns for ticker extraction from model names:
    // - lstm_model_AAPL (directory) -> AAPL
    // - rf_model_TSLA.joblib -> TSLA  
    // - xgboost_model_MSFT.pkl -> MSFT
    // - model_GOOGL_20231201.h5 -> GOOGL
    
    const patterns = [
        /^lstm_model_([A-Z]{1,5})$/i,  // LSTM directory format: lstm_model_TICKER
        /_([A-Z]{1,5})\.(?:keras|joblib|pkl|h5)$/i,  // Standard format: model_TICKER.extension
        /_([A-Z]{1,5})_\d+\.(?:keras|joblib|pkl|h5)$/i,  // With date: model_TICKER_date.extension
        /([A-Z]{1,5})_model/i,  // Prefix format: TICKER_model
        /_([A-Z]{1,5})$/i,  // Simple underscore format: whatever_TICKER
        /([A-Z]{2,5})(?:_|\.|$)/i  // Any uppercase 2-5 letters followed by separator or end
    ];
      for (const pattern of patterns) {
        const match = modelName.match(pattern);
        if (match && match[1]) {
            return match[1].toUpperCase();
        }
    }
    
    return null;
};

// Helper function to format file size
const formatFileSize = (sizeInMB) => {
    if (!sizeInMB || isNaN(sizeInMB)) return 'Desconocido';
    
    const size = parseFloat(sizeInMB);
    
    if (size >= 1024) {
        return `${(size / 1024).toFixed(2)} GB`;
    } else if (size >= 1) {
        return `${size.toFixed(2)} MB`;
    } else {
        return `${(size * 1024).toFixed(0)} KB`;
    }
};

function ConfigurationPanel({
    availableModelTypes,
    availableTickers,
    onConfigChange, 
    onTrain,
    onForecast,
    initialConfig,
    dateRangeWarning,
    trainingStatus,
    currentTrainingJob,
    currentJobId,
    isPollingStatus,
    trainingStatusMessage,
    trainMutationPending,
    forecastMutationPending,    pollingError,
    isCurrentConfigurationTraining = false,  // New prop
    activeTrainingJobs = [],  // New prop for showing active jobs
    availableModels = null,  // Available models data for current model type
    availableModelsLoading = false,  // Loading state for available models
    availableModelsError = null,  // Error state for available models
    // New props for loading model data
    onLoadModelData = null,  // Callback to load model data when clicking on trained models
    isLoadingModelData = false,  // Loading state for model data operations
    loadedModelError = null  // Error state for model loading operations
}) {
    // Estado local para manejar los valores del formulario
    // Lo inicializamos desde initialConfig para que refleje el estado de App.js
    const [formValues, setFormValues] = useState({
         ...initialConfig,
         dateRange: [dateToDayjs(initialConfig.startDate), dateToDayjs(initialConfig.endDate)],
         customDateRange: [dateToDayjs(initialConfig.custom_start_date), dateToDayjs(initialConfig.custom_end_date)],
         targetCol: 'Close', // Fixed value - always Close
     });
       // Estados para el modal de detalles del modelo
    const [modelDetailsVisible, setModelDetailsVisible] = useState(false);
    const [selectedModelDetails, setSelectedModelDetails] = useState(null);
    
    // Estado para mostrar "Encolando..." brevemente
    const [isEnqueuing, setIsEnqueuing] = useState(false);

    useEffect(() => {
         setFormValues({
             ...initialConfig,
             dateRange: [dateToDayjs(initialConfig.startDate), dateToDayjs(initialConfig.endDate)],
             customDateRange: [dateToDayjs(initialConfig.custom_start_date), dateToDayjs(initialConfig.custom_end_date)],
             targetCol: 'Close', // Fixed value - always Close
         });
     }, [initialConfig]);const handleValueChange = (changedValues) => {
        const newValues = { ...formValues, ...changedValues };

        // Always ensure targetCol is 'Close'
        newValues.targetCol = 'Close';

        // Manejar cambios en el RangePicker principal (mantener compatibilidad)
        if ('dateRange' in changedValues) {
            const [startDayjs, endDayjs] = changedValues.dateRange || [null, null];
            newValues.startDate = dayjsToDate(startDayjs);
            newValues.endDate = dayjsToDate(endDayjs);
        }

        // Manejar cambios en el preset de período de entrenamiento
        if ('training_period_preset' in changedValues) {
            const preset = changedValues.training_period_preset;
            
            if (preset === 'custom') {
                // Si se selecciona custom, limpiar las fechas automáticas y usar las personalizadas
                newValues.training_period_preset = 'custom';
                newValues.custom_start_date = formValues.custom_start_date || null;
                newValues.custom_end_date = formValues.custom_end_date || null;
            } else {
                // Si se selecciona un preset, limpiar las fechas personalizadas
                newValues.training_period_preset = preset;
                newValues.custom_start_date = null;
                newValues.custom_end_date = null;
                
                // Calcular fechas basadas en el preset seleccionado
                const endDate = new Date();
                let startDate = new Date();
                
                switch (preset) {
                    case '1_year':
                        startDate.setFullYear(endDate.getFullYear() - 1);
                        break;
                    case '3_years':
                        startDate.setFullYear(endDate.getFullYear() - 3);
                        break;
                    case '5_years':
                        startDate.setFullYear(endDate.getFullYear() - 5);
                        break;
                    default:
                        startDate.setFullYear(endDate.getFullYear() - 3); // Default a 3 años
                }
                
                newValues.startDate = startDate;
                newValues.endDate = endDate;
                newValues.dateRange = [dateToDayjs(startDate), dateToDayjs(endDate)];
            }
        }

        // Manejar cambios en el RangePicker personalizado
        if ('customDateRange' in changedValues) {
            const [startDayjs, endDayjs] = changedValues.customDateRange || [null, null];
            newValues.custom_start_date = dayjsToDate(startDayjs);
            newValues.custom_end_date = dayjsToDate(endDayjs);
            
            // Si hay fechas personalizadas válidas, actualizamos también las fechas principales
            if (startDayjs && endDayjs) {
                newValues.startDate = dayjsToDate(startDayjs);
                newValues.endDate = dayjsToDate(endDayjs);
                newValues.dateRange = [startDayjs, endDayjs];
            }
        }

        setFormValues(newValues); // Actualizar estado local
        onConfigChange(newValues); // Notificar al padre con el objeto completo actualizado
    };    // Manejador para el botón de Entrenar - Enhanced for non-blocking behavior
    const handleTrainClick = async () => {
        // Show "Enqueuing..." state briefly
        setIsEnqueuing(true);
        
        try {
            // Call the parent's training function
            await onTrain(formValues.selectedModelType, formValues);
            
            // Show brief success feedback
            message.success('Entrenamiento encolado exitosamente', 2);
        } catch (error) {
            // Error handling will be done in the parent component
            console.warn('Training submission error:', error);
        } finally {
            // Always reset the enqueuing state after a brief delay
            setTimeout(() => {
                setIsEnqueuing(false);
            }, 1500); // Reset after 1.5 seconds
        }
    };

    // Manejador para el botón de Pronosticar
     const handleForecastClick = () => {
         onForecast(formValues.selectedModelType, formValues);
     };    // Handler for model click (auto-select ticker and load model data)
    const handleModelClick = async (model) => {
        const extractedTicker = extractTickerFromModelName(model.name || model.filename);
        
        if (extractedTicker && extractedTicker !== formValues.selectedTicker) {
            // Update the ticker in form values
            const updatedValues = {
                ...formValues,
                selectedTicker: extractedTicker
            };
            setFormValues(updatedValues);
            onConfigChange(updatedValues);
            
            message.success(`Ticker "${extractedTicker}" seleccionado automáticamente`);
        } else if (extractedTicker === formValues.selectedTicker) {
            message.info(`El ticker "${extractedTicker}" ya está seleccionado`);
        } else {
            message.warning('No se pudo extraer el ticker del nombre del modelo');
        }

        // Load model data if callback is available
        if (onLoadModelData && extractedTicker) {
            try {
                await onLoadModelData(formValues.selectedModelType, model.name || model.filename, extractedTicker);
            } catch (error) {
                console.error('Error loading model data from ConfigurationPanel:', error);
            }
        }
    };

    // Handler for showing model details
    const handleShowModelDetails = (model, e) => {
        e.stopPropagation(); // Prevent triggering the model click
        setSelectedModelDetails(model);
        setModelDetailsVisible(true);
    };

    // Handler for retraining a model
    const handleRetrainModel = (model, e) => {
        e.stopPropagation(); // Prevent triggering the model click
        
        const extractedTicker = extractTickerFromModelName(model.name || model.filename);
        const modelType = formValues.selectedModelType;
        
        Modal.confirm({
            title: 'Re-entrenar Modelo',
            content: `¿Estás seguro de que quieres re-entrenar el modelo "${model.name || model.filename}"${extractedTicker ? ` para el ticker ${extractedTicker}` : ''}?`,
            okText: 'Re-entrenar',
            cancelText: 'Cancelar',
            onOk: () => {
                // If ticker is different, update it first
                if (extractedTicker && extractedTicker !== formValues.selectedTicker) {
                    const updatedValues = {
                        ...formValues,
                        selectedTicker: extractedTicker
                    };
                    setFormValues(updatedValues);
                    onConfigChange(updatedValues);
                    
                    // Trigger training with updated config
                    setTimeout(() => {
                        onTrain(modelType, updatedValues);
                        message.info(`Iniciando re-entrenamiento del modelo ${modelType.toUpperCase()} para ${extractedTicker}...`);
                    }, 100);
                } else {
                    // Use current config
                    onTrain(modelType, formValues);
                    message.info(`Iniciando re-entrenamiento del modelo ${modelType.toUpperCase()}...`);
                }
            },
        });
    };

    // Handler for deleting a model
    const handleDeleteModel = (model, e) => {
        e.stopPropagation(); // Prevent triggering the model click
        
        Modal.confirm({
            title: 'Eliminar Modelo',
            content: `¿Estás seguro de que quieres eliminar el modelo "${model.name || model.filename}"? Esta acción no se puede deshacer.`,
            okText: 'Eliminar',
            cancelText: 'Cancelar',
            okType: 'danger',
            onOk: () => {
                // TODO: Implement actual model deletion API call
                message.info('Función de eliminación de modelos próximamente disponible');
                // For now, just show a message
                console.log('Delete model:', model);
            },
        });
    };

    // Layout para los items del formulario (label arriba, input abajo)
    const formItemLayout = {
        labelCol: { span: 24 },
        wrapperCol: { span: 24 },
    };    return (
        <Form
            {...formItemLayout}
             layout="vertical" // Labels arriba
             onValuesChange={handleValueChange} // Llama a handleValueChange cuando cualquier campo cambia
             initialValues={{ // Sincronizar valores iniciales 
                 selectedTicker: formValues.selectedTicker,
                 dateRange: formValues.dateRange,
                 selectedModelType: formValues.selectedModelType,
                 nLags: formValues.nLags,
                 forecastHorizon: formValues.forecastHorizon,
                 training_period_preset: formValues.training_period_preset || '3_years',
                 customDateRange: formValues.customDateRange
             }}
        >
            {/* Selección de Ticker */}
            <Form.Item label="Acción (Ticker)" name="selectedTicker">
                <Select placeholder="Selecciona un ticker">
                    {availableTickers.map(ticker => <Option key={ticker} value={ticker}>{ticker}</Option>)}
                </Select>
            </Form.Item>            {/* Selección de Modelo */}
            <Form.Item label="Tipo de Modelo" name="selectedModelType">
                 <Select placeholder="Selecciona un tipo de modelo">
                     {availableModelTypes.map(type => <Option key={type} value={type}>{type.toUpperCase()}</Option>)}
                 </Select>
            </Form.Item>

            {/* Modelos Disponibles - Enhanced version */}
            {formValues.selectedModelType && (
                <Form.Item label={`Modelos Entrenados (${formValues.selectedModelType?.toUpperCase()})`}>
                    {availableModelsLoading ? (
                        <div style={{ textAlign: 'center', padding: '16px' }}>
                            <Spin size="small" />
                            <span style={{ marginLeft: '8px', color: '#666' }}>Cargando modelos...</span>
                        </div>
                    ) : availableModelsError ? (
                        <Alert
                            message="Error al cargar modelos"
                            description={availableModelsError}
                            type="warning"
                            showIcon
                            size="small"
                        />
                    ) : availableModels && availableModels.total_models > 0 ? (
                        <div style={{ 
                            background: '#f6ffed', 
                            border: '1px solid #b7eb8f', 
                            borderRadius: '6px', 
                            padding: '12px',
                            fontSize: '13px'
                        }}>
                            <div style={{ 
                                color: '#52c41a', 
                                fontWeight: 'bold', 
                                marginBottom: '12px',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'space-between'
                            }}>
                                <span>✅ {availableModels.total_models} modelo{availableModels.total_models > 1 ? 's' : ''} disponible{availableModels.total_models > 1 ? 's' : ''}</span>
                                <span style={{ fontSize: '10px', color: '#666' }}>Haz clic para auto-seleccionar ticker</span>
                            </div>

                            {availableModels.models && availableModels.models.length > 0 && (
                                <div style={{ color: '#666' }}>
                                    <div style={{ marginBottom: '8px', fontWeight: '500' }}>
                                        Modelos disponibles:
                                    </div>
                                    {availableModels.models.map((model, index) => {
                                        const extractedTicker = extractTickerFromModelName(model.name || model.filename);
                                        const isCurrentTicker = extractedTicker === formValues.selectedTicker;
                                        const formattedMetadata = formatMetadata(model.metadata);
                                        
                                        return (
                                            <Tooltip
                                                key={index}
                                                title={
                                                    <div style={{ maxWidth: '300px' }}>
                                                        <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>
                                                            {model.name || model.filename || 'Modelo sin nombre'}
                                                        </div>
                                                        {extractedTicker && (
                                                            <div style={{ marginBottom: '4px' }}>
                                                                <strong>Ticker:</strong> {extractedTicker}
                                                            </div>
                                                        )}
                                                        {model.size_mb && (
                                                            <div style={{ marginBottom: '4px' }}>
                                                                <strong>Tamaño:</strong> {formatFileSize(model.size_mb)}
                                                            </div>
                                                        )}
                                                        {formattedMetadata && formattedMetadata !== 'No disponible' && (
                                                            <div>
                                                                <strong>Última actualización:</strong> {
                                                                    formattedMetadata.timestamp || 
                                                                    formattedMetadata.created_at || 
                                                                    'No disponible'
                                                                }
                                                            </div>
                                                        )}
                                                        <div style={{ marginTop: '6px', fontSize: '10px', opacity: 0.8 }}>
                                                            💡 Haz clic para auto-seleccionar ticker
                                                        </div>
                                                    </div>
                                                }
                                                placement="right"
                                            >
                                                <div 
                                                    key={index} 
                                                    onClick={() => handleModelClick(model)}
                                                    style={{ 
                                                        marginBottom: index < availableModels.models.length - 1 ? '8px' : '0',
                                                        paddingLeft: '8px',
                                                        paddingRight: '8px',
                                                        paddingTop: '6px',
                                                        paddingBottom: '6px',
                                                        borderLeft: `3px solid ${isCurrentTicker ? '#1890ff' : '#d9f7be'}`,
                                                        backgroundColor: isCurrentTicker ? '#e6f7ff' : 'transparent',
                                                        borderRadius: '4px',
                                                        cursor: 'pointer',
                                                        transition: 'all 0.2s ease',
                                                        position: 'relative'
                                                    }}
                                                    onMouseEnter={(e) => {
                                                        e.target.style.backgroundColor = isCurrentTicker ? '#bae7ff' : '#f0f9f0';
                                                        e.target.style.transform = 'translateX(2px)';
                                                    }}
                                                    onMouseLeave={(e) => {
                                                        e.target.style.backgroundColor = isCurrentTicker ? '#e6f7ff' : 'transparent';
                                                        e.target.style.transform = 'translateX(0)';
                                                    }}
                                                >
                                                    <div style={{ 
                                                        display: 'flex', 
                                                        justifyContent: 'space-between', 
                                                        alignItems: 'center' 
                                                    }}>
                                                        <div style={{ flex: 1 }}>
                                                            <div style={{ 
                                                                fontFamily: 'monospace', 
                                                                fontSize: '11px', 
                                                                fontWeight: '500',
                                                                display: 'flex',
                                                                alignItems: 'center',
                                                                gap: '6px'
                                                            }}>
                                                                📁 {model.name || model.filename || 'Modelo sin nombre'}
                                                                {isCurrentTicker && (
                                                                    <Tag size="small" color="blue">actual</Tag>
                                                                )}
                                                                {extractedTicker && (
                                                                    <Tag size="small" color="green">{extractedTicker}</Tag>
                                                                )}
                                                            </div>
                                                            
                                                            <div style={{ display: 'flex', gap: '8px', marginTop: '2px' }}>
                                                                {model.size_mb && (
                                                                    <div style={{ fontSize: '10px', color: '#999' }}>
                                                                        📊 {formatFileSize(model.size_mb)}
                                                                    </div>
                                                                )}
                                                                {formattedMetadata && formattedMetadata !== 'No disponible' && (
                                                                    <div style={{ fontSize: '10px', color: '#666' }}>
                                                                        ℹ️ Con metadatos
                                                                    </div>
                                                                )}
                                                            </div>
                                                        </div>
                                                        
                                                        {/* Action buttons */}
                                                        <div style={{ display: 'flex', gap: '4px' }}>
                                                            <Button
                                                                type="text"
                                                                size="small"
                                                                icon={<EyeOutlined />}
                                                                onClick={(e) => handleShowModelDetails(model, e)}
                                                                style={{ fontSize: '10px' }}
                                                                title="Ver detalles"
                                                            />
                                                            <Button
                                                                type="text"
                                                                size="small"
                                                                icon={<ReloadOutlined />}
                                                                onClick={(e) => handleRetrainModel(model, e)}
                                                                style={{ fontSize: '10px' }}
                                                                title="Re-entrenar modelo"
                                                            />
                                                            <Button
                                                                type="text"
                                                                size="small"
                                                                danger
                                                                icon={<DeleteOutlined />}
                                                                onClick={(e) => handleDeleteModel(model, e)}
                                                                style={{ fontSize: '10px' }}
                                                                title="Eliminar modelo"
                                                            />
                                                        </div>
                                                    </div>
                                                </div>
                                            </Tooltip>
                                        );
                                    })}
                                </div>
                            )}
                        </div>
                    ) : (
                        <Alert
                            message="No hay modelos entrenados"
                            description={`No se encontraron modelos entrenados para ${formValues.selectedModelType?.toUpperCase()}. Entrena un modelo primero.`}
                            type="info"
                            showIcon
                            size="small"
                        />
                    )}                </Form.Item>
            )}

            {/* Loading state for model data operations */}
            {isLoadingModelData && (
                <Form.Item>
                    <Alert
                        message={
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <Spin size="small" />
                                <span>Cargando datos del modelo seleccionado...</span>
                            </div>
                        }
                        type="info"
                        showIcon={false}
                        style={{ marginBottom: '16px' }}
                    />
                </Form.Item>
            )}

            {/* Error state for model data loading */}
            {loadedModelError && (
                <Form.Item>
                    <Alert
                        message="Error al cargar datos del modelo"
                        description={loadedModelError}
                        type="error"
                        showIcon
                        style={{ marginBottom: '16px' }}
                    />
                </Form.Item>
            )}

            {/* Model Details Modal */}
            <Modal
                title="Detalles del Modelo"
                open={modelDetailsVisible}
                onCancel={() => setModelDetailsVisible(false)}
                footer={[
                    <Button key="close" onClick={() => setModelDetailsVisible(false)}>
                        Cerrar
                    </Button>
                ]}
                width={600}
            >
                {selectedModelDetails && (
                    <div>
                        <Descriptions
                            bordered
                            size="small"
                            column={1}
                            items={[
                                {
                                    key: 'name',
                                    label: 'Nombre del Archivo',
                                    children: selectedModelDetails.name || selectedModelDetails.filename || 'Sin nombre'
                                },
                                {
                                    key: 'ticker',
                                    label: 'Ticker Detectado',
                                    children: extractTickerFromModelName(selectedModelDetails.name || selectedModelDetails.filename) || 'No detectado'
                                },                                {
                                    key: 'status',
                                    label: 'Estado',
                                    children: <Tag color="success">Activo</Tag>
                                },
                                {
                                    key: 'size',
                                    label: 'Tamaño del Archivo',
                                    children: selectedModelDetails.size_mb ? formatFileSize(selectedModelDetails.size_mb) : 'No disponible'
                                },
                                {
                                    key: 'metadata',
                                    label: 'Metadatos',
                                    children: (() => {
                                        const metadata = formatMetadata(selectedModelDetails.metadata);
                                        if (metadata === 'No disponible' || !metadata) {
                                            return 'No disponible';
                                        }
                                        
                                        if (typeof metadata === 'object') {
                                            return (
                                                <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                                                    <pre style={{ 
                                                        fontSize: '11px', 
                                                        margin: 0, 
                                                        whiteSpace: 'pre-wrap',
                                                        wordBreak: 'break-word'
                                                    }}>
                                                        {JSON.stringify(metadata, null, 2)}
                                                    </pre>
                                                </div>
                                            );
                                        }
                                        
                                        return String(metadata);
                                    })()
                                }
                            ]}
                        />
                        
                        <div style={{ marginTop: '16px', padding: '12px', backgroundColor: '#f0f9f0', borderRadius: '6px' }}>
                            <Space direction="vertical" size="small">
                                <div style={{ fontWeight: 'bold', color: '#52c41a' }}>💡 Acciones Disponibles:</div>
                                <div style={{ fontSize: '12px' }}>
                                    • <strong>Auto-seleccionar:</strong> Haz clic en el modelo para seleccionar automáticamente el ticker correspondiente
                                </div>
                                <div style={{ fontSize: '12px' }}>
                                    • <strong>Ver detalles:</strong> Usa el botón 👁️ para ver esta información detallada
                                </div>
                                <div style={{ fontSize: '12px' }}>
                                    • <strong>Re-entrenar:</strong> Usa el botón 🔄 para entrenar el modelo con datos actualizados
                                </div>
                                <div style={{ fontSize: '12px' }}>
                                    • <strong>Eliminar:</strong> Usa el botón 🗑️ para eliminar el modelo (próximamente)
                                </div>
                            </Space>
                        </div>
                    </div>
                )}
            </Modal>

            <Divider /> {/* Línea divisoria */}            <Title level={3} style={{ color: '#1890ff', marginBottom: '16px' }}>📈 Generar Pronóstico</Title> {/* Título principal para pronóstico */}

            {/* Horizonte de Pronóstico */}
            <Form.Item label="Horizonte de Pronóstico (días)" name="forecastHorizon">
                <InputNumber min={1} step={1} style={{ width: '100%' }} />
            </Form.Item>            {/* Botón para Generar Pronóstico - ACCIÓN PRINCIPAL */}
             <Form.Item>
                 <Button
                    type="primary" // Botón principal
                    size="large" // Tamaño más grande para mayor prominencia
                    onClick={handleForecastClick}
                    block
                    loading={forecastMutationPending} // Show spinner while generating forecast
                    disabled={
                        !formValues.selectedTicker || 
                        !formValues.selectedModelType ||
                        forecastMutationPending ||
                        isCurrentConfigurationTraining  // Only disable if THIS config is training
                    } // Allow forecasting while other models train
                    style={{ marginBottom: '24px' }} // Espacio adicional después del botón principal
                 >
                    {forecastMutationPending 
                        ? '🔄 Generando pronóstico...' 
                        : `📊 Generar Pronóstico (${formValues.selectedModelType?.toUpperCase()})`
                    }
                </Button>
            </Form.Item>

            <Divider style={{ margin: '32px 0' }} /> {/* Línea divisoria más prominente */}

            <Title level={4} style={{ color: '#8c8c8c' }}>⚙️ Gestión de Modelos (Avanzado)</Title> {/* Título secundario para entrenamiento */}

            {/* Selección de Período de Entrenamiento */}
            <Form.Item label="Período de Entrenamiento" name="training_period_preset">
                <Select placeholder="Selecciona el período">
                    <Option value="1_year">Último año</Option>
                    <Option value="3_years">Últimos 3 años</Option>
                    <Option value="5_years">Últimos 5 años</Option>
                    <Option value="custom">Rango Personalizado</Option>
                </Select>
            </Form.Item>

            {/* RangePicker Personalizado - Solo visible cuando se selecciona "custom" */}
            {formValues.training_period_preset === 'custom' && (
                <Form.Item label="Rango Fechas Personalizado" name="customDateRange">
                    <RangePicker style={{ width: '100%' }} format="YYYY-MM-DD" />
                </Form.Item>
            )}

            {/* Selección de Rango de Fechas - Solo para mostrar las fechas actuales */}
            <Form.Item label="Rango Fechas Actual (Solo Lectura)" name="dateRange">
                <RangePicker 
                    style={{ width: '100%' }} 
                    format="YYYY-MM-DD" 
                    disabled={true}
                    placeholder={['Fecha inicio', 'Fecha fin']}
                />
            </Form.Item>

            {/* Date Range Warning */}
            {dateRangeWarning && (
                <Form.Item>
                    <Alert
                        message={dateRangeWarning}
                        type="warning"
                        showIcon
                        style={{ marginBottom: '16px' }}
                    />
                </Form.Item>
            )}

            {/* Número de Lags */}
            <Form.Item label="Número de Lags (Entrenamiento)" name="nLags">
                <InputNumber min={1} step={1} style={{ width: '100%' }} />
            </Form.Item>            {/* Botón de Entrenamiento - ACCIÓN SECUNDARIA */}
            <Form.Item> {/* Wrapper para el botón */}
                <Button
                    type="dashed" // Estilo menos prominente
                    onClick={handleTrainClick}
                    block // Ocupa todo el ancho
                    loading={isEnqueuing || trainMutationPending} // Show spinner while submitting or enqueuing
                    disabled={
                        !formValues.selectedTicker || 
                        !formValues.selectedModelType || 
                        isEnqueuing ||
                        trainMutationPending ||
                        isCurrentConfigurationTraining  // Only disable if THIS config is actively training
                    }
                >
                    {isEnqueuing 
                        ? '📤 Encolando...'
                        : trainMutationPending 
                            ? 'Enviando solicitud...' 
                            : isCurrentConfigurationTraining 
                                ? `🔄 Entrenando ${formValues.selectedModelType?.toUpperCase()}...`
                                : `🎯 Entrenar/Actualizar Modelo (${formValues.selectedModelType?.toUpperCase()})`
                    }
                </Button>
            </Form.Item>{/* Status message and polling indicator - only for current configuration */}
            {isCurrentConfigurationTraining && (isPollingStatus || trainingStatusMessage) && (
                <Form.Item>
                    <Alert
                        message={
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                {isPollingStatus && <Spin size="small" />}
                                <span>
                                    {trainingStatusMessage || 'Procesando entrenamiento...'}
                                </span>
                            </div>
                        }
                        type={
                            trainingStatus === 'failed' ? 'error' :
                            trainingStatus === 'completed' ? 'success' :
                            'info'
                        }
                        showIcon={false}
                        style={{ marginBottom: '16px' }}
                    />
                </Form.Item>
            )}

            {/* Error display - only for current configuration */}
            {isCurrentConfigurationTraining && pollingError && (
                <Form.Item>
                    <Alert
                        message="Error en el entrenamiento"
                        description={pollingError}
                        type="error"
                        showIcon
                        style={{ marginBottom: '16px' }}
                    />
                </Form.Item>
            )}
        </Form>
    );
}

export default ConfigurationPanel;