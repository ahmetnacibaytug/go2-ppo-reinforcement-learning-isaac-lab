# Unitree Go2: PPO ile hız takibi

Isaac Lab üzerinde Unitree Go2 dört ayaklı robotu için pekiştirmeli öğrenme ile hız takibi çalışması. PPO politikası; ileri, geri, yanal hareket, dönüş ve durma komutlarına göre robotun eklem hedeflerini üretir. Proje eğitim ve değerlendirme kodunu, eğitilmiş model dosyalarını ve örnek videoları içerir.

> Bu çalışma simülasyon içindir. Gerçek bir Go2 üzerinde test edildiği iddia edilmemektedir. ROS 2 veya Gazebo entegrasyonu bu depoda henüz uygulanmamıştır.

## Gösterim

- [Yürüyüş videosu — 1080p, 60 fps](media/go2_yuruyus_1080p60.mp4)
- [Hareket testi — 1080p, 30 fps](media/Robot_Hareket_Testi.mp4)
- [Çalışma raporu](docs/Reinforcement_Learning_Case_Study.pdf)

## Yaklaşım

- **Ortam:** düz zemin üzerinde Unitree Go2; Isaac Lab 2.3.2 ve Isaac Sim 5.1.
- **Algoritma:** RSL-RL ile Proximal Policy Optimization (PPO); raporda kullanılan sürüm RSL-RL 3.1.2'dir.
- **Girdiler:** gövde doğrusal ve açısal hızları, izdüşürülmüş yerçekimi, hız komutu, göreli eklem konumları ve hızları, önceki aksiyon. Raporda toplam 48 gözlem bileşeni belirtilir.
- **Çıktılar:** 12 eklem için pozisyon aksiyonu; ortam yapılandırmasında 0,25 ölçeği ve varsayılan eklem konumu ofseti kullanılır.
- **Zamanlama:** 5 ms fizik adımı (200 Hz) ve dört fizik adımında bir politika güncellemesi (50 Hz).
- **Dayanıklılık:** sürtünme, gövde kütlesi ve aktüatör kazançları rastgeleleştirilir; gözlem gürültüsü ve dış itmeler yapılandırılabilir. İsteğe bağlı sol–sağ simetri veri artırımı da bulunur.

## Raporlanan sonuçlar

Çalışma raporunda, son modelin 10 seed ve seed başına 16 ortamda değerlendirildiği; her senaryoda 500 kontrol adımı çalıştırıldığı ve ilk 50 adımın RMSE hesabı dışında tutulduğu belirtiliyor. Aşağıdaki değerler **rapordan aktarılmıştır**; bu depoda ham değerlendirme kayıtları yer almadığı için burada bağımsız olarak yeniden doğrulanmamıştır.

| Komut | XY RMSE (m/s) | Yaw RMSE (rad/s) | Raporlanan düşme |
|---|---:|---:|---:|
| İleri, 0,8 m/s | 0,056 | 0,093 | 0 / 160 |
| Geri, −0,4 m/s | 0,095 | 0,101 | 0 / 160 |
| Yanal, 0,5 m/s | 0,067 | 0,101 | 0 / 160 |
| Dönüş, +0,7 rad/s | 0,054 | 0,143 | 0 / 160 |
| Dönüş, −0,7 rad/s | 0,053 | 0,133 | 0 / 160 |
| Durma | 0,028 | 0,085 | 0 / 160 |

Düşme sayısı raporda testin tamamı için verilmiştir. Bu değerler belirtilen simülasyon senaryolarına aittir; gerçek robot performansı olarak yorumlanmamalıdır. Ayrıntılar ve değerlendirme tanımları için [rapora](docs/Reinforcement_Learning_Case_Study.pdf) bakın.

## Gereksinimler

- [Isaac Lab 2.3.2 kurulum rehberi](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html) ve uyumlu Isaac Sim 5.1 ortamı.
- Isaac Lab'in RSL-RL entegrasyonu ve `rsl-rl-lib` 3.x. Proje kodu 3.0.1 veya üzerini, 4.0.0 altını kontrol eder; raporda 3.1.2 kullanılmıştır.
- Kurulumun sağladığı Python, PyTorch ve Gymnasium bağımlılıkları. İsteğe bağlı grafik veya video işlemleri için Matplotlib ve/veya FFmpeg gerekebilir.
- Uyumlu bir NVIDIA GPU. Eğitim ve RTX görselleştirme için kaynak ihtiyacı senaryoya göre değişir; resmi kurulum rehberindeki gereksinimleri kontrol edin.

## Eğitilmiş modeli deneme

Aşağıdaki örnek Linux içindir. Isaac Lab kurulumunuzun kök dizinine geçin ve `PROJECT_DIR` değerini bu deponun **mutlak yolu** ile değiştirin. Komutlar hazır bir Isaac Lab ortamı varsayar.

```bash
cd /path/to/IsaacLab
export PROJECT_DIR=/path/to/go2-ppo
./isaaclab.sh -p "$PROJECT_DIR/kod/play.py" \
  --checkpoint "$PROJECT_DIR/model/model_899.pt" \
  --num_envs 1 --cmd 0.8 0.0 0.0 --no_video --no_export --no_plot
```

Bu örnek ileri yönde 0,8 m/s komutu verir. `--cmd` sırasıyla ileri hız (m/s), yanal hız (m/s) ve dönüş hızıdır (rad/s). Değerlendirme çıktıları checkpoint'in bulunduğu klasörde `evaluation/` altına yazılır. Depodaki hazır modelin yeniden üretilmesi ve örnek komutlar bu bilgisayarda Isaac Lab ile çalıştırılarak doğrulanmamıştır.

Farklı komutları topluca değerlendirmek için `--cmd 0.8 0.0 0.0` yerine `--suite` kullanabilirsiniz. `--suite` ve `--cmd` birlikte kullanılmaz; sağa dönüşü ve ek dönüş hızlarını eklemek için `--suite --turn_sweep` kullanın. Raporlanan değerlendirme düzenine yaklaşmak için ayrıca `--steps 500 --warmup 50 --num_envs 16 --robustness` seçilebilir. Rapordaki 10 seed'i yeniden üretmek için bu koşuyu her seed için ayrı çalıştırıp sonuçları ayrıca birleştirmek gerekir; tek komut rapordaki tabloyu üretmez.

## Yeni eğitim

Isaac Lab kök dizininde ve yukarıdaki `PROJECT_DIR` ayarlanmışken:

```bash
./isaaclab.sh -p "$PROJECT_DIR/kod/train.py" --preset smoke --headless
```

`smoke`, kısa bir kurulum kontrolü için 128 ortam ve 10 iterasyon kullanır. Eğitim çıktıları varsayılan olarak `kod/logs/` altında tutulur. `budget`, `balanced` ve `full` seçenekleri daha büyük eğitimler içindir. Sol–sağ simetri veri artırımı `--symmetry` ile açılabilir; mevcut bir eğitimi sürdürmek için `--resume /path/to/model.pt` kullanılır.

## Dosya yapısı

```text
kod/       Eğitim, değerlendirme, ortam ve yardımcı Python dosyaları
model/     RSL-RL checkpoint, TorchScript ve ONNX politika dosyaları
docs/      Çalışma raporu
media/     Simülasyon videoları
```

`model_899.pt` değerlendirme betiğinin kullandığı RSL-RL checkpoint'idir. `policy.pt` ve `policy.onnx` dışa aktarılmış politika dosyalarıdır; bunların başka bir simülatörde kullanılabilmesi için gözlem sırası, normalizasyon, eklem sırası ve aksiyon ölçeğinin eşleşmesi gerekir.

## Sınırlar ve sonraki adımlar

- Politika Isaac Lab ortamı için geliştirilmiştir. Gazebo'ya veya fiziksel Go2'ye aktarım ayrıca doğrulanmalıdır.
- Değerlendirme raporu ve videoları dahildir; ham seed bazlı sonuç dosyaları ve eğitim logları bu pakette yoktur.
- ROS 2, Gazebo ve canlı sensör entegrasyonu planlanan çalışmalardır; bu repoda hazır değildir.

## Atıflar ve lisans

Çalışma NVIDIA Isaac Lab / Isaac Sim, RSL-RL ve Unitree Go2 varlıklarını kullanır. Bu araçlar ve varlıklar kendi lisans ve kullanım koşullarına tabidir. Bu proje için ayrıca bir açık kaynak lisansı henüz seçilmemiştir.
