/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { UIAgentMessage } from '#modules/messages/types.ts';
import { getMessageForm } from '#modules/messages/utils.ts';
import { useAgentRun } from '#modules/runs/contexts/agent-run/index.ts';
import { useTask } from '#modules/tasks/contexts/task-context/index.ts';
import { blurActiveElement } from '#utils/dom-utils.ts';

import type { RunFormValues } from '../types';
import { FormRenderer } from './FormRenderer';

interface Props {
  message: UIAgentMessage;
}

export function MessageForm({ message }: Props) {
  const formPart = getMessageForm(message);
  const { submitForm } = useAgentRun();
  const { task } = useTask();

  const isLastMessage = task.messages.at(-1)?.id === message.id;

  if (!formPart) {
    return null;
  }

  return (
    <FormRenderer
      definition={formPart}
      showHeading={false}
      isDisabled={!isLastMessage}
      onSubmit={(values: RunFormValues) => {
        const form = {
          request: formPart,
          response: { id: formPart.id, values },
        };

        submitForm(form, task.id);

        blurActiveElement();
      }}
    />
  );
}
