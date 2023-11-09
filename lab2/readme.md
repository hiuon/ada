2. Yolo - развернуть, посмотреть картинки, отчет по использованию

### Использовал данную реализацию: https://github.com/hank-ai/darknet
---
### Далее был долгий процесс установки darknet и всех зависимостей (для работы на GPU с использованием различных библиотек от nvidia):
![Screenshot](./pictures/install.png)
---
### Я решил попробовать определять на картинках где Том, а где Джерри из Тома и Джерри. Набор картинок нашел [тут](https://www.kaggle.com/datasets/balabaskar/tom-and-jerry-image-classification). С помощью дополнительных инструментов разметил несколько десятков картинок:
![Screenshot](./pictures/mark.png)
---
### После разметки, начал процесс обучения (заняло некторое время):
![Screenshot](./pictures/train.png)

![Screenshot](./pictures/end_train.png)
---
### После, используя данные обучения проверил детектор на других картинках:
![Screenshot](./pictures/result1.png)
![Screenshot](./pictures/result2.png)
![Screenshot](./pictures/result3.png)
![Screenshot](./pictures/result4.png)
---
### Уже как-то может определять тестируемые объекты:).