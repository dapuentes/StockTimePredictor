import { useMutation } from '@tanstack/react-query';
import { trainModel, generateForecast } from '../services/api';
import { message } from 'antd'; 

// Hook para la mutación de entrenamiento
/**
 * Custom hook to handle the mutation for training a machine learning model.
 * 
 * This hook uses `useMutation` to manage the API call for training a model and provides
 * lifecycle callbacks for handling loading, success, error, and settled states.
 * 
 * @param {Object} [options={}] - Optional configuration for the mutation.
 * @param {Function} [options.onSuccessCallback] - Callback function to execute on successful mutation.
 * @param {Function} [options.onErrorCallback] - Callback function to execute on mutation error.
 * @param {Function} [options.onSettledCallback] - Callback function to execute when the mutation is settled (either success or error).
 * @param {...Object} [options] - Additional options to pass to the `useMutation` hook.
 * 
 * @returns {Object} - The mutation object returned by `useMutation`, including methods like `mutate` and `mutateAsync`.
 * 
 * @example
 * const { mutate } = useTrainModelMutation({
 *   onSuccessCallback: (data, variables) => {
 *     console.log('Model trained successfully:', data);
 *   },
 *   onErrorCallback: (error) => {
 *     console.error('Error training model:', error);
 *   },
 * });
 * 
 * mutate({ modelType: 'neural-network', config: { epochs: 10 } });
 */
export function useTrainModelMutation(options = {}) {
    return useMutation({
        mutationFn: ({ modelType, config }) => trainModel(modelType, config), // La función que llama a la API

        onMutate: () => {

            message.loading({ content: 'Entrenando...', key: 'trainStatus', duration: 0 });
        },
        onSuccess: (data, variables, context) => {

            message.success({ content: `Modelo ${variables.modelType.toUpperCase()} entrenado!`, key: 'trainStatus', duration: 3 });
            if (options.onSuccessCallback) options.onSuccessCallback(data, variables); // Llama a callback si se proporcionó
        },
        onError: (error, variables, context) => {

            message.error({ content: `Error entrenando: ${error.message}`, key: 'trainStatus', duration: 5 });
             if (options.onErrorCallback) options.onErrorCallback(error, variables); // Llama a callback si se proporcionó
        },
        onSettled: (data, error, variables, context) => {

             message.destroy('trainStatus'); // Asegura cerrar el mensaje loading
             if (options.onSettledCallback) options.onSettledCallback(data, error, variables);
         },
        ...options // Permite pasar opciones adicionales al hook
    });
}

// Hook para la mutación de pronóstico
/**
 * Custom hook to handle the generation of forecasts using a mutation.
 *
 * @param {Object} [options={}] - Optional configuration for the mutation.
 * @param {Function} [options.onSuccessCallback] - Callback function to execute on successful mutation.
 * @param {Function} [options.onErrorCallback] - Callback function to execute on mutation error.
 * @param {Function} [options.onSettledCallback] - Callback function to execute when the mutation is settled (either success or error).
 * @returns {Object} - The mutation object returned by `useMutation`.
 *
 * @example
 * const { mutate, isLoading } = useGenerateForecastMutation({
 *   onSuccessCallback: (data) => console.log('Forecast generated:', data),
 *   onErrorCallback: (error) => console.error('Error generating forecast:', error),
 *   onSettledCallback: () => console.log('Mutation settled'),
 * });
 *
 * mutate({ modelType: 'linear', config: { param1: 'value1' } });
 */
export function useGenerateForecastMutation(options = {}) {
     return useMutation({
         mutationFn: ({ modelType, config }) => generateForecast(modelType, config),
         onMutate: () => {
             message.loading({ content: 'Generando pronóstico...', key: 'forecastStatus', duration: 0 });
         },
         onSuccess: (data, variables, context) => {
             message.success({ content: 'Pronóstico generado!', key: 'forecastStatus', duration: 3 });
             if (options.onSuccessCallback) options.onSuccessCallback(data);
         },
         onError: (error, variables, context) => {
             message.error({ content: `Error pronosticando: ${error.message}`, key: 'forecastStatus', duration: 5 });
             if (options.onErrorCallback) options.onErrorCallback(error);
         },
         onSettled: (data, error, variables, context) => {
             message.destroy('forecastStatus');
             if (options.onSettledCallback) options.onSettledCallback();
         },
         ...options
     });
 }