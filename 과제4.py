import hashlib
import time
import sys
import collections

# ==========================================
# 1. 다중 해시 함수 생성기 (Salting 활용)
# ==========================================
def get_hashes(item, num_hashes, max_val):
    """하나의 아이템에 대해 서로 다른 num_hashes 개의 해시 값을 생성"""
    hashes = []
    item_str = str(item)
    for i in range(num_hashes):
        # 각 차수마다 다른 salt를 부여하여 독립성 확보
        encoded = f"{item_str}-{i}".encode('utf-8')
        h = int(hashlib.md5(encoded).hexdigest(), 16)
        hashes.append(h % max_val)
    return hashes

# ==========================================
# 2. Bloom Filter 구현
# ==========================================
class BloomFilter:
    def __init__(self, size, num_hashes):
        self.size = size
        self.num_hashes = num_hashes
        self.bit_array = bytearray(size)  # 메모리 효율을 위해 bytearray 사용

    def add(self, item):
        for h in get_hashes(item, self.num_hashes, self.size):
            self.bit_array[h] = 1

    def __contains__(self, item):
        for h in get_hashes(item, self.num_hashes, self.size):
            if self.bit_array[h] == 0:
                return False
        return True

# ==========================================
# 3. Count-Min Sketch 구현
# ==========================================
class CountMinSketch:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = [[0] * width for _ in range(depth)]

    def add(self, item, count=1):
        for d in range(self.depth):
            encoded = f"{item}-{d}".encode('utf-8')
            h = int(hashlib.md5(encoded).hexdigest(), 16) % self.width
            self.table[d][h] += count

    def estimate(self, item):
        min_val = float('inf')
        for d in range(self.depth):
            encoded = f"{item}-{d}".encode('utf-8')
            h = int(hashlib.md5(encoded).hexdigest(), 16) % self.width
            min_val = min(min_val, self.table[d][h])
        return min_val

# ==========================================
# 4. 데이터 스트림 시뮬레이터 (Line-by-line)
# ==========================================
def data_stream_generator(file_path):
    with open(file_path, 'r', encoding='latin-1') as f:
        for line in f:
            if line.strip():
                # UserID::MovieID::Rating::Timestamp
                parts = line.strip().split('::')
                if len(parts) >= 2:
                    yield parts[1]  # MovieID 추출

# ==========================================
# 5. 실험 및 성능 측정 메인 루프
# ==========================================
def run_experiment(file_path, bf_size, bf_hashes, cms_width, cms_depth):
    print(f"\n====== 실험 시작 (BF_Size: {bf_size}, CMS_Width: {cms_width}) ======")
    
    # 구조 초기화
    bf = BloomFilter(size=bf_size, num_hashes=bf_hashes)
    cms = CountMinSketch(width=cms_width, depth=cms_depth)
    
    # Ground Truth 계산용 내장 자료구조
    gt_set = set()
    gt_counter = collections.Counter()
    
    # 스트림 처리 및 시간 측정
    start_time = time.time()
    total_records = 0
    
    for movie_id in data_stream_generator(file_path):
        total_records += 1
        
        # 알고리즘 반영
        bf.add(movie_id)
        cms.add(movie_id)
        
        # Ground Truth 업데이트
        gt_set.add(movie_id)
        gt_counter[movie_id] += 1
        
    elapsed_time = time.time() - start_time
    throughput = total_records / elapsed_time
    
    # ------------------------------------------
    # 결과 평가 및 정확도 산출
    # ------------------------------------------
    # 1. Bloom Filter False Positive Rate (FPR) 측정
    # 스트림에 등장하지 않은 가상의 MovieID 5,000개를 대상으로 테스트
    fp_count = 0
    negative_samples = [f"not_exist_movie_{i}" for i in range(5000)]
    for sample in negative_samples:
        if sample in bf:
            fp_count += 1
    fpr = (fp_count / len(negative_samples)) * 100
    
    # 2. Count-Min Sketch 정확도 측정 (평균 절대 오차 - MAE)
    total_error = 0
    for movie_id, actual_count in gt_counter.items():
        est_count = cms.estimate(movie_id)
        total_error += abs(est_count - actual_count)
    mae = total_error / len(gt_counter)
    
    # 3. 메모리 측정 (가상 이론적 크기 계산)
    bf_mem_kb = bf_size / 8 / 1024  # bits to KB
    cms_mem_kb = (cms_width * cms_depth * 4) / 1024  # 4-byte integer 가정
    
    print(f"처리된 총 레코드 수: {total_records:,} 개")
    print(f"전체 처리 시간: {elapsed_time:.2f} 초 (초당 {throughput:.2f} 레코드)")
    print(f"[Bloom Filter] FPR: {fpr:.2f}% | 할당 메모리: {bf_mem_kb:.2f} KB")
    print(f"[Count-Min Sketch] MAE (평균 오차): {mae:.2f} | 할당 메모리: {cms_mem_kb:.2f} KB")
    
    return {
        "fpr": fpr, "mae": mae, "time": elapsed_time, 
        "bf_mem": bf_mem_kb, "cms_mem": cms_mem_kb
    }

if __name__ == "__main__":
    # 파일 경로 지정 (사용자 환경에 맞게 변경 가능)
    dataset_path = r"C:\Users\fbwls\Desktop\ratings.dat"
    
    # 파라미터 비교 실험 진행
    # Case A: 적은 메모리 설정
    results_A = run_experiment(dataset_path, bf_size=5000, bf_hashes=3, cms_width=500, cms_depth=3)
    
    # Case B: 넉넉한 메모리 설정
    results_B = run_experiment(dataset_path, bf_size=50000, bf_hashes=5, cms_width=5000, cms_depth=5)