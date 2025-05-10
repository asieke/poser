import {
  DEFAULT_CHAT_MODELS,
  DEFAULT_PROVIDERS,
} from '../../../constants';
import { SettingMigration, SmartComposerSettings } from '../setting.types';
import {
  getMigratedChatModels,
  getMigratedProviders,
  ExistingSettingsData,
} from './migrationUtils';
import { LLMProvider } from '../../../types/provider.types'; // For casting
import { ChatModel } from '../../../types/chat-model.types'; // For casting

/**
 * Migration from version 10 to version 11
 * - Add Sieke LLM provider
 * - Add Sieke LLM default chat model
 */

export const migrateFrom10To11: SettingMigration['migrate'] = (
  data: ExistingSettingsData,
) => {
  const newData: Partial<SmartComposerSettings> & { version: number } = {
    ...data,
    version: 11
  } as any;

  // Pass the full default providers and models from constants.ts
  // The migration utils should handle merging these correctly with existing settings.
  // The types DefaultProviders and DefaultChatModels in migrationUtils are for the *shape* of the default list argument.
  newData.providers = getMigratedProviders(data, DEFAULT_PROVIDERS as any) as LLMProvider[];
  newData.chatModels = getMigratedChatModels(data, DEFAULT_CHAT_MODELS as any) as ChatModel[];

  const siekeDefaultChatModel = (newData.chatModels as ChatModel[]).find(m => m.providerType === 'sieke' && m.id === 'sieke-default-chat');
  const fallbackChatModelId = (newData.chatModels as ChatModel[])[0]?.id || ''
  const siekeDefaultChatModelId = siekeDefaultChatModel?.id || fallbackChatModelId;

  let currentChatModelId = newData.chatModelId;
  const currentModelIsValid = (newData.chatModels as ChatModel[]).find(m => m.id === currentChatModelId);

  if (!currentChatModelId || !currentModelIsValid) {
    newData.chatModelId = siekeDefaultChatModelId;
  }

  if (newData.applyModelId) {
    const applyModelIsValid = (newData.chatModels as ChatModel[]).find(m => m.id === newData.applyModelId);
    if (!applyModelIsValid) {
      newData.applyModelId = siekeDefaultChatModelId;
    }
  } else {
    newData.applyModelId = siekeDefaultChatModelId;
  }

  return newData as SmartComposerSettings;
}; 