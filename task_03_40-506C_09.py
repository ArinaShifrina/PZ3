import numpy 
from numpy.fft import fft, fftshift
import matplotlib.pyplot as plt
import tools

class WaveletRicker:
    '''
    Источник, создающий импульс в форме вейвлета Рикера
    '''

    def __init__(self, Np, Md, eps=1.0, mu=1.0, Sc=1.0, magnitude=1.0):
        '''
        Nl - количество отсчетов на длину волны;
        Md - определяет задержку импульса;
        eps - относительная д/э проницаемость, в которой находится источник;
        mu - относительная магнитная проницаемость, в которой находится источник;
        Sc - число Куранта;
        magnitude - максимальное значение в источнике.
        '''
        self.Np = Np
        self.Md = Md
        self.eps = eps
        self.mu = mu
        self.Sc = Sc
        self.magnitude = magnitude

    def getField(self, m, q):
        t = (numpy.pi ** 2) * (self.Sc *
            (q - m * numpy.sqrt(self.eps * self.mu)) / self.Np - self.Md) ** 2
        return self.magnitude * (1 - 2 * t) * numpy.exp(-t)
    

if __name__ == '__main__':
    # Волновое сопротивление свободного пространства
    W0 = 120.0 * numpy.pi

    # Число Куранта
    Sc = 1.0

    # Скорость света
    c = 3e8

    # Время расчета в отсчетах
    maxTime = 1500

    # Размер области моделирования вдоль оси X в метрах
    X = 4.5

    #Размер ячейки разбиения
    dx = 1e-2

    # Размер области моделирования в отсчетах
    maxSize = int(X / dx)

    #Шаг дискретизации по времени
    dt = Sc * dx / c

    # Положение источника в отсчетах
    sourcePos = int(maxSize / 2)

    # Датчики для регистрации поля
    probesPos = [sourcePos - 75]
    probes = [tools.Probe(pos, maxTime) for pos in probesPos]

    # Параметры среды
    # Диэлектрическая проницаемость
    eps = numpy.ones(maxSize)
    eps[:] = 1.5

    # Магнитная проницаемость
    mu = numpy.ones(maxSize)

    Ez = numpy.zeros(maxSize)
    Hy = numpy.zeros(maxSize)

    source = WaveletRicker(35.0, 2.5, eps[sourcePos], mu[sourcePos])

    # Ez[-2] В предыдущий момент времени
    oldEzRight = Ez[-2]

    # Расчет коэффициентов для граничных условий
    tempRight = Sc / numpy.sqrt(mu[-1] * eps[-1])
    koeffABCRight = (tempRight - 1) / (tempRight + 1)
    
    # Параметры отображения поля E
    display_field = Ez
    display_ylabel = 'Ez, В/м'
    display_ymin = -1.1
    display_ymax = 1.1

    # Создание экземпляра класса для отображения
    # распределения поля в пространстве
    display = tools.AnimateFieldDisplay(maxSize,
                                        display_ymin, display_ymax,
                                        display_ylabel, dx)

    display.activate()
    display.drawProbes(probesPos)
    display.drawSources([sourcePos])

    for q in range(maxTime):
        # Граничные условия для поля Н (слева)
        Hy[0] = 0
        
        # Расчет компоненты поля H
        Ez_shift = Ez[:-1]
        Hy[1:] = Hy[1:] + (Ez_shift - Ez[1:]) * Sc / (W0 * mu[1:])

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        # Создается источник, бегущий влево, который отразится от 
        # левой стенки согласно заданию 9 варианта (граничные условия PMC - коэф-т
        # отражения электрической компоненты +1) и затем побежит вправо.
        Hy[sourcePos] -= Sc / (W0 * mu[sourcePos]) * source.getField(0, q)

        # Расчет компоненты поля E
        Hy_shift = Hy[1:]
        Ez[:-1] = Ez[:-1] + (Hy[:-1] - Hy_shift) * Sc * W0 / eps[:-1]

        # Источник возбуждения с использованием метода
        # Total Field / Scattered Field
        Ez[sourcePos - 1] += (Sc / (numpy.sqrt(eps[sourcePos - 1] * mu[sourcePos - 1])) *
                          source.getField(-0.5, q + 0.5))

        # Граничные условия АВС первой степени (справа)
        Ez[-1] = oldEzRight + koeffABCRight * (Ez[-2] - Ez[-1])
        oldEzRight = Ez[-2]

        # Регистрация поля в датчиках
        for probe in probes:
            probe.addData(Ez, Hy)

        if q % 5 == 0:
            display.updateData(display_field, q)

    display.stop()

    # Отображение сигнала, сохраненного в датчиках
    tools.showProbeSignals(probes, -1.1, 1.1, dt)

    # Отображение спектра сигнала
    spectrum = tools.Spectrum(probe.E, dt, 3e9)
    spectrum.fourierTransform()

