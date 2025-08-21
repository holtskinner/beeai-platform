/**
 * Copyright 2025 © BeeAI a Series of LF Projects, LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type { FormRender } from '#api/a2a/extensions/ui/form.ts';
import { Container } from '#components/layouts/Container.tsx';
import { FormRenderer } from '#modules/form/components/FormRenderer.tsx';
import type { RunFormValues } from '#modules/form/types.ts';

import { useAgentRun } from '../contexts/agent-run';
import classes from './RunLandingView.module.scss';

interface Props {
  formRender: FormRender;
}

export function RunFormView({ formRender }: Props) {
  const { agent } = useAgentRun();

  if (!formRender) {
    return false;
  }

  return (
    <Container size="sm" className={classes.root}>
      <FormRenderer
        definition={formRender}
        onSubmit={(values: RunFormValues) => {
          console.log(values);
        }}
        defaultTitle={agent.ui.user_greeting}
      />
    </Container>
  );
}
