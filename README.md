# Camera-accuracy-studies
Investigation of how the accuracy of recognizing object positions on the camera changes under physical influences  <br />
Many detectors выделяет поле по границе и отобрвжвет вид сверху. Не использует модуль hyper_params  <br />
Что важно для алгоритма:  <br />
Расположение камеры с центрального ракурса и захват самой дальней и самой близкой линий(линии поля около ворот)  <br />
Есть проблема в неверном распознавании вертикальных линий, если они сильно закрыты <br />
