import os
import json
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import Select

class BatteryTestParser:
    BASE_URL = "https://batterytest.ru"
    RESULTS_URL = f"{BASE_URL}/res"
    API_GRAPH_URL = f"{BASE_URL}/api/get_data_graph_from_db.php"

    def __init__(self, driver_path=None, headless=True, output_dir="yan_stud_dataset"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")
        if driver_path:
            self.driver = webdriver.Chrome(executable_path=driver_path, options=options)
        else:
            self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 15)

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'X-Requested-With': 'XMLHttpRequest',
        })

    def _get_test_ids_from_results(self, types=None, sizes=None, chems=None):
        """Собирает test_id и brand_model с главной страницы, выбирая отображение всех записей."""
        params = []
        if types:
            params.append(f"types={','.join(types)}")
        if sizes:
            params.append(f"sizes={','.join(sizes)}")
        if chems:
            params.append(f"chems={','.join(chems)}")
        url = self.RESULTS_URL
        if params:
            url += "?" + "&".join(params)
        print(f"Открываем страницу результатов: {url}")
        self.driver.get(url)

        # Нажимаем "Показать", если кнопка есть
        try:
            show_btn = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'Показать')]"))
            )
            show_btn.click()
            print("Кнопка 'Показать' нажата.")
            time.sleep(3)
        except TimeoutException:
            print("Кнопка 'Показать' не найдена — таблица уже загружена или фильтры не требуют подтверждения.")

        # Выбираем "Все" в выпадающем списке количества записей
        try:
            select_elem = self.wait.until(
                EC.presence_of_element_located((By.NAME, "res_length"))
            )
            select = Select(select_elem)
            select.select_by_value("-1")  # значение "Все"
            print("Выбрано отображение всех записей.")
            time.sleep(3)  # ждём перезагрузку таблицы
        except Exception as e:
            print(f"Не удалось выбрать 'Все': {e}")

        # Ждём таблицу
        try:
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "table#res tbody tr")))
        except TimeoutException:
            print("Таблица с результатами не загрузилась.")
            return []

        rows = self.driver.find_elements(By.CSS_SELECTOR, "table#res tbody tr")
        tests = []
        for row in rows:
            try:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 5:
                    continue
                brand_elem = cells[1].find_element(By.TAG_NAME, "a")
                brand_model = brand_elem.text.strip()
                chart_btn = cells[2].find_element(By.CLASS_NAME, "btn_chart")
                test_id = chart_btn.get_attribute("data-no")
                if test_id:
                    tests.append({'test_id': test_id, 'brand_model': brand_model})
            except Exception:
                continue
        print(f"Найдено тестов: {len(tests)}")
        return tests

    def _parse_test_page(self, test_id):
        """Парсит страницу /{test_id} и возвращает полные данные."""
        url = f"{self.BASE_URL}/{test_id}"
        resp = self.session.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')

        info = {}
        h1 = soup.find('h1')
        if h1:
            info['full_title'] = h1.text.strip()

        detail_table = soup.find('table', class_='table-bordered')
        if detail_table:
            for row in detail_table.find_all('tr'):
                cells = row.find_all('td')
                if len(cells) >= 2:
                    key = cells[0].text.strip().rstrip(':')
                    value = cells[1].text.strip()
                    info[key] = value

        measurements = []
        results_table = soup.find('table', class_='monospace_font_body')
        if not results_table:
            return {'info': info, 'measurements': measurements}

        headers = []
        thead = results_table.find('thead')
        if thead:
            header_rows = thead.find_all('tr')
            last_header_row = header_rows[-1] if header_rows else None
            if last_header_row:
                for th in last_header_row.find_all(['th', 'td']):
                    headers.append(th.text.strip())

        tbody = results_table.find('tbody')
        if not tbody:
            return {'info': info, 'measurements': measurements}

        rows = tbody.find_all('tr')
        pending_spans = {}

        for row_idx, row in enumerate(rows):
            cells = row.find_all(['td', 'th'])
            row_data = {}
            col_idx = 0
            cell_idx = 0

            while col_idx < len(headers):
                if col_idx in pending_spans:
                    span_info = pending_spans[col_idx]
                    row_data[headers[col_idx]] = span_info['value']
                    span_info['remaining'] -= 1
                    if span_info['remaining'] == 0:
                        del pending_spans[col_idx]
                    col_idx += 1
                    continue

                if cell_idx >= len(cells):
                    break

                cell = cells[cell_idx]
                cell_text = cell.text.strip()
                rowspan = int(cell.get('rowspan', 1))
                colspan = int(cell.get('colspan', 1))

                for offset in range(colspan):
                    current_col = col_idx + offset
                    if current_col < len(headers):
                        row_data[headers[current_col]] = cell_text
                        if rowspan > 1:
                            pending_spans[current_col] = {
                                'value': cell_text,
                                'remaining': rowspan - 1
                            }

                cell_idx += 1
                col_idx += colspan

            if row_data:
                measurements.append(row_data)

        for m in measurements:
            for k, v in m.items():
                if isinstance(v, str) and v.replace('.', '', 1).isdigit():
                    m[k] = float(v) if '.' in v else int(v)

        return {'info': info, 'measurements': measurements}

    def _fetch_graph_data(self, test_id, mode=1):
        payload = {'no': test_id, 'mode': mode}
        try:
            resp = self.session.post(self.API_GRAPH_URL, data=payload, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  [!] Ошибка API графика для {test_id}: {e}")
            return None

    def collect_data(self, types=None, sizes=None, chems=None, filename="battery_full_data.csv"):
        filepath = os.path.join(self.output_dir, filename)
        if os.path.exists(filepath):
            print(f"Файл '{filepath}' уже существует. Загружаем из кэша.")
            return pd.read_csv(filepath)

        tests = self._get_test_ids_from_results(types, sizes, chems)
        if not tests:
            print("Не удалось получить список тестов.")
            return pd.DataFrame()

        self.driver.quit()
        print("Selenium завершён, переходим к парсингу страниц...")

        all_data = []
        for idx, test in enumerate(tests, 1):
            test_id = test['test_id']
            print(f"[{idx}/{len(tests)}] Обработка теста {test_id}...")

            try:
                page_data = self._parse_test_page(test_id)
            except Exception as e:
                print(f"  Ошибка парсинга страницы {test_id}: {e}")
                page_data = {'info': {}, 'measurements': []}

            graph = self._fetch_graph_data(test_id)

            # Извлекаем удобные поля из графика (time, ua, ub)
            discharge_curve = None
            if graph:
                # Структура: {"data1": {"time": [...], "ua": [...], "ub": [...]}, "data2": {...}}
                data1 = graph.get('data1', {})
                discharge_curve = {
                    'time': data1.get('time', []),
                    'ua': data1.get('ua', []),
                    'ub': data1.get('ub', []),
                    'mode1': graph.get('mode1', ''),
                    'mode2': graph.get('mode2', '')
                }

            record = {
                'test_id': test_id,
                'brand_model': test['brand_model'],
                'info': json.dumps(page_data['info'], ensure_ascii=False),
                'measurements': json.dumps(page_data['measurements'], ensure_ascii=False),
                'discharge_curve': json.dumps(discharge_curve, ensure_ascii=False) if discharge_curve else None
            }
            all_data.append(record)
            time.sleep(0.5)

        df = pd.DataFrame(all_data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Собрано {len(df)} записей. Данные сохранены в '{filepath}'.")
        return df


if __name__ == "__main__":
    parser = BatteryTestParser(headless=True)  # headless=True для фонового режима
    df = parser.collect_data(
        types=['accu', 'bat'],   # все аккумуляторы и батарейки
        # sizes=['AA'],           # при необходимости раскомментировать
        # chems=['Lithium'],
        filename="all_batteries.csv"
    )
    if not df.empty:
        print("\nПример первой записи:")
        sample = df.iloc[0].to_dict()
        for k, v in sample.items():
            if k in ['info', 'measurements', 'discharge_curve']:
                print(f"  {k}: <JSON строка>")
            else:
                print(f"  {k}: {v}")
