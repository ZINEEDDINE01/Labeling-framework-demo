import { Translation } from "@ngx-translate/core";


export interface AppTranslation extends Translation {
  title: string;
}

export const translations: AppTranslation = {
  title: 'demo.title',
};
