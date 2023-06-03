-- phpMyAdmin SQL Dump
-- version 5.2.0
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: May 28, 2023 at 08:57 AM
-- Server version: 10.4.24-MariaDB
-- PHP Version: 7.4.29

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `web_service`
--

-- --------------------------------------------------------

--
-- Table structure for table `face`
--

CREATE TABLE `face` (
  `id` int(12) NOT NULL,
  `jenis_kelamin` varchar(25) NOT NULL,
  `rentang_umur` varchar(25) NOT NULL,
  `label` varchar(10) NOT NULL,
  `akurasi` varchar(8) NOT NULL,
  `nama_file` text NOT NULL,
  `date_created` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `face`
--

INSERT INTO `face` (`id`, `jenis_kelamin`, `rentang_umur`, `label`, `akurasi`, `nama_file`, `date_created`) VALUES
(4, 'Laki-Laki', '>=20 Tahun', 'Dewasa', '99.85', 'save/Image/sampel4.jpg', '2023-05-26 16:03:25'),
(5, 'Perempuan', '6-11 Tahun', 'Anak', '99.81', 'save/Image/anakanak.jpg', '2023-05-26 16:06:19'),
(6, 'Laki-Laki', '>=20 Tahun', 'Dewasa', '99.06', 'save/Image/test.jpg', '2023-05-27 03:46:02'),
(7, 'Laki-Laki', '12-19 Tahun', 'Remaja', '59.79', 'save/Image/testing.jpg', '2023-05-27 03:54:58'),
(8, 'Laki-Laki', '>=20 Tahun', 'Dewasa', '99.06', 'save/Image/test.jpg', '2023-05-27 03:55:16'),
(9, 'Perempuan', '12-19 Tahun', 'Remaja', '92.57', 'save/Image/download.jpg', '2023-05-27 03:55:30'),
(10, 'Perempuan', '>=20 Tahun', 'Dewasa', '96.33', 'save/Image/test1.jpg', '2023-05-27 03:56:19'),
(11, 'Perempuan', '>=20 Tahun', 'Dewasa', '96.33', 'save/Image/test1.jpg', '2023-05-27 04:04:54');

-- --------------------------------------------------------

--
-- Table structure for table `gender`
--

CREATE TABLE `gender` (
  `id` int(11) NOT NULL,
  `jenis_kelamin` varchar(100) NOT NULL,
  `rentang_umur` varchar(100) NOT NULL,
  `label` varchar(100) DEFAULT NULL,
  `akurasi` varchar(8) NOT NULL,
  `nama_file` text NOT NULL,
  `date_created` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `gender`
--

INSERT INTO `gender` (`id`, `jenis_kelamin`, `rentang_umur`, `label`, `akurasi`, `nama_file`, `date_created`) VALUES
(76, 'Laki-Laki', '6-11 Tahun', 'Anak', '100.0', 'save/Audio/Anak-L-996.wav', '2023-05-26 15:44:00'),
(77, 'Perempuan', '>=20 Tahun', 'Dewasa', '86.45', 'save/Audio/Dewasa-P-999.wav', '2023-05-26 15:44:17'),
(78, 'Laki-Laki', '6-11 Tahun', 'Anak', '99.68', 'save/Audio/Anak-L-0.wav', '2023-05-26 16:09:50'),
(79, 'Laki-Laki', '6-11 Tahun', 'Anak', '99.68', 'save/Audio/Anak-L-0.wav', '2023-05-26 16:09:57'),
(80, 'Laki-Laki', '12-19 Tahun', 'Remaja', '90.25', 'save/Audio/Remaja-L-1.wav', '2023-05-26 16:10:16'),
(81, 'Perempuan', '12-19 Tahun', 'Remaja', '99.8', 'save/Audio/Remaja-P-5.wav', '2023-05-26 16:10:27'),
(82, 'Laki-Laki', '>=20 Tahun', 'Dewasa', '82.27', 'save/Audio/Dewasa-L-1.wav', '2023-05-26 16:10:44'),
(83, 'Laki-Laki', '12-19 Tahun', 'Remaja', '89.57', 'save/Audio/tesAnak.wav', '2023-05-27 03:46:25'),
(84, 'Perempuan', '12-19 Tahun', 'Remaja', '53.96', 'save/Audio/tes-fiqih.wav', '2023-05-27 03:46:43'),
(85, 'Perempuan', '12-19 Tahun', 'Remaja', '96.59', 'save/Audio/cobatest.wav', '2023-05-27 03:49:09'),
(86, 'Laki-Laki', '>=20 Tahun', 'Dewasa', '96.47', 'save/Audio/Dewasa-L-16.wav', '2023-05-27 03:50:00'),
(87, 'Laki-Laki', '>=20 Tahun', 'Dewasa', '82.84', 'save/Audio/Hallo_DewasaL.wav', '2023-05-27 03:50:24'),
(88, 'Perempuan', '12-19 Tahun', 'Remaja', '54.08', 'save/Audio/Hallo_DewasaP.wav', '2023-05-27 03:51:17'),
(89, 'Perempuan', '12-19 Tahun', 'Remaja', '61.89', 'save/Audio/Hallo_RemajaLP.wav', '2023-05-27 03:51:39'),
(90, 'Laki-Laki', '12-19 Tahun', 'Remaja', '89.57', 'save/Audio/tesAnak.wav', '2023-05-27 03:52:17'),
(91, 'Perempuan', '12-19 Tahun', 'Remaja', '53.96', 'save/Audio/tes-fiqih.wav', '2023-05-27 03:52:37'),
(92, 'Perempuan', '>=20 Tahun', 'Dewasa', '86.45', 'save/Audio/Dewasa-P-999.wav', '2023-05-27 11:34:54'),
(93, 'Laki-Laki', '12-19 Tahun', 'Remaja', '90.25', 'save/Audio/Remaja-L-1.wav', '2023-05-28 02:33:18'),
(94, 'Laki-Laki', '>=20 Tahun', 'Dewasa', '82.27', 'save/Audio/Dewasa-L-1.wav', '2023-05-28 02:33:34'),
(95, 'Laki-Laki', '12-19 Tahun', 'Remaja', '89.57', 'save/Audio/tesAnak.wav', '2023-05-28 02:33:51'),
(96, 'Perempuan', '12-19 Tahun', 'Remaja', '96.59', 'save/Audio/cobatest.wav', '2023-05-28 02:35:27'),
(97, 'Laki-Laki', '6-11 Tahun', 'Anak', '100.0', 'save/Audio/Anak-L-996.wav', '2023-05-28 02:36:17'),
(98, 'Laki-Laki', '6-11 Tahun', 'Anak', '99.68', 'save/Audio/Anak-L-0.wav', '2023-05-28 02:36:59'),
(99, 'Perempuan', '12-19 Tahun', 'Remaja', '99.8', 'save/Audio/Remaja-P-5.wav', '2023-05-28 03:07:50'),
(100, 'Laki-Laki', '6-11 Tahun', 'Anak', '100.0', 'save/Audio/Anak-L-996.wav', '2023-05-28 03:09:22'),
(101, 'Laki-Laki', '>=20 Tahun', 'Dewasa', '96.47', 'save/Audio/Dewasa-L-16.wav', '2023-05-28 03:09:51');

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `email` varchar(100) NOT NULL,
  `name` varchar(100) NOT NULL,
  `password` varchar(256) NOT NULL,
  `token` text DEFAULT NULL,
  `status_validasi` text DEFAULT NULL,
  `level` varchar(256) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp(),
  `update_at` timestamp NOT NULL DEFAULT current_timestamp() ON UPDATE current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `email`, `name`, `password`, `token`, `status_validasi`, `level`, `created_at`, `update_at`) VALUES
(3, 'mohfiqiherinsyah@gmail.com', 'Moh. Fiqih Erinsyah', 'pbkdf2:sha256:260000$mMbBJPQP1TWAr2Ub$b69bd608348847afda72cae32c735c270c150756389750643b72cbfbed28e43d', 'bW9oZmlxaWhlcmluc3lhaEBnbWFpbC5jb20=', 'Valid', 'Administrator', '2023-04-25 03:56:20', '2023-05-20 02:16:10'),
(9, 'mfiqiherinsyah90@gmail.com', 'mfiqiherinsyah90', 'pbkdf2:sha256:600000$aBxd1n0tbw7UV5lK$1e1e22b3c65871644ed604dac7c326865e86aa1dc73325530e7759681b35693a', 'bWZpcWloZXJpbnN5YWg5MEBnbWFpbC5jb20=', 'Valid', 'User', '2023-05-08 08:31:51', '2023-05-19 13:16:57'),
(10, 'vitakarenina06@gmail.com', 'Vita Karenina', 'pbkdf2:sha256:600000$9Rf4kRVETy2SJcDR$a6d632caaaed7a6e99dfc7e81011fa7728b01c1bc98afba943a7cd0f87482f33', NULL, 'Belum Validasi', 'Administrator', '2023-05-08 08:33:01', '2023-05-15 14:52:51'),
(20, 'mfiqiher@gmail.com', 'm fiqih', 'pbkdf2:sha256:600000$STLU9K30Xo8VwZ49$9de1efb3833625f7b1bcfca35c7e943f0d40178fe9126558d33323f1c7421878', NULL, 'Belum Validasi', 'Administrator', '2023-05-25 15:48:46', '2023-05-25 15:48:46');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `face`
--
ALTER TABLE `face`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `gender`
--
ALTER TABLE `gender`
  ADD PRIMARY KEY (`id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `email` (`email`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `face`
--
ALTER TABLE `face`
  MODIFY `id` int(12) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=12;

--
-- AUTO_INCREMENT for table `gender`
--
ALTER TABLE `gender`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=102;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=21;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
