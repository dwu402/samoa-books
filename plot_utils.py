from matplotlib import pyplot as plt
from matplotlib import font_manager
from matplotlib import dates as mdates
import datetime

label_font = font_manager.FontProperties(
    family=['cmr10'],
    weight='regular',
    size=16,
)

legend_font = font_manager.FontProperties(
    family=['cmr10'],
    weight='regular',
    size=14,
)

tick_font = font_manager.FontProperties(
    family=['cmr10'],
    weight='regular',
    size=12,
)

def set_font():
    plt.rcParams['figure.constrained_layout.use'] = True

    plt.rcParams.update(
        {
            'text.usetex': False,
            'font.family': 'cmr10',
            'font.serif': 'cmr10',
            'mathtext.fontset': 'cm',
            "mathtext.default": 'regular',
        }
    )

xdate_format = mdates.DateFormatter("1 %b %Y")
at_months = mdates.MonthLocator()
at_half_month = mdates.DayLocator([15])

def form_xmonths(ax):
    ax.xaxis.set_major_locator(at_months)
    ax.xaxis.set_major_formatter(xdate_format)
    ax.xaxis.set_minor_locator(at_half_month)
