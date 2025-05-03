import { useMutation } from '@tanstack/react-query';
import { trainModel, generateForecast } from '../services/api';
import { message } from 'antd';

// Hook para la mutación de entrenamiento
/**
 * Custom hook to create a mutation for training a model. It manages the state and logic around
 * the training process, including API calls, loading messages, and error handling.
 *
 * @param {Object} [options={}] Optional parameters to configure the mutation behavior.
 * @param {Function} [options.onSuccessCallback] Callback function executed after successful mutation.
 * @param {Function} [options.onErrorCallback] Callback function executed after a failed mutation.
 * @param {Function} [options.onSettledCallback] Callback function executed after the mutation completes, regardless of success or error.
 * @param {Object} [options.otherOptions] Additional options to customize the mutation.
 *
 * @return {Object} Returns an object from `useMutation` hook that includes methods and state for executing and tracking the mutation.
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
 * Custom hook for triggering a forecast generation mutation.
 *
 * @param {Object} [options={}] Configuration options for customizing the mutation behavior.
 * @param {Function} [options.onSuccessCallback] Optional callback invoked upon a successful mutation.
 * @param {Function} [options.onErrorCallback] Optional callback invoked if the mutation fails.
 * @param {Function} [options.onSettledCallback] Optional callback invoked when the mutation is settled, whether successful or failed.
 *
 * @return {Object} Returns the mutation object with functions and state, allowing you to trigger the mutation and monitor its status.
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